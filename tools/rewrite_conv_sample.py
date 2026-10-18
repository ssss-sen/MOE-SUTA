# -*- coding: utf-8 -*-
"""把 MoE 卷积专家（及其它逐块路由）重写为样本级实现，并自带验收。

与"外科手术式插入"的区别
  - 直接替换整个方法体：moe_conv.MoEConv1dLayer.get_topks / forward
  - 一并重写：moe_model.MoETTA.get_coeff_topks、moe_normalization._get_topks_block、
              moe_proj.MoELinearLayer.get_topks
  - 重写后这些文件里**不再出现** num_blocks / unfold / F_out_chunk / 逐块池化（静态断言，按文件整体检查）

不变量
  1) 配置文件/环境变量一律不动
  2) 快/慢两路融合口径与更新范围不动（fast=topk 单专家、slow=prob 全专家）
  3) 路由数值与重写前逐位一致 ⇒ 用 _t_conv_equiv3.py 验收（期望 max|Δ| == 0）
  4) 提交节奏（steps/update_freq/memory）不动

用法
  python3 rewrite_conv_sample.py --dryrun     # 只检查（锚点 + 静态断言 + 编译），不写文件
  python3 rewrite_conv_sample.py              # 执行（自动存参考 → 重写 → 自动跑等价测试）
"""
import io
import os
import shutil
import subprocess
import sys
import time
import tokenize
import py_compile


def code_only(src):
    """去掉注释与字符串后的代码 token 串（用于"代码里不许再有块标识"的断言）。"""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            out.append(tok.string)
    except Exception:
        return src
    return " ".join(out)

NEW_GET_TOPKS = '''    def get_topks(self, x: torch.Tensor):
        """逐样本路由：整条音频只做一次专家决策。

        统计量 = 音频每秒的均值再对秒数求平均（与历史逐位一致，故结果可比）；
        决策/系数形状一律 [B, K] / [B, E]。
        x shape: [B, in_C, L]
        """
        B, in_C, L = x.shape
        fps = max(1, int(self.frames_per_sec))
        nb = math.ceil(L / fps)
        pad_len = (fps - (L % fps)) % fps
        x_pad_route = F.pad(x, (0, pad_len))
        chunk_means = x_pad_route.view(B, in_C, nb, fps).mean(dim=-1).transpose(1, 2)   # [B, nb, in_C]
        flat = chunk_means.mean(dim=1)                                                  # [B, in_C] <- 每样本一个统计量

        route_in = flat
        if getattr(self, "batch_route", False):
            cur = flat.detach().mean(0, keepdim=True)
            self._br_acc = cur if self._br_acc is None else self._br_acc + cur
            self._br_n += 1
            if self._br_ema is not None:
                route_in = self._br_ema.expand(flat.shape[0], -1)

        prob = F.softmax(self.router(route_in), dim=-1)          # [B, E]
        biased = prob - self.penalty
        fe = getattr(self, "force_expert", None)
        if fe is not None:
            topks = torch.full((biased.size(0), self.topk), int(fe), dtype=torch.long, device=biased.device)
            vals = biased.gather(1, topks)
        elif getattr(self, "force_random", False):
            topks = torch.randint(0, self.num_expert, (biased.size(0), self.topk), device=biased.device)
            vals = biased.gather(1, topks)
        else:
            vals, topks = torch.topk(biased, self.topk, dim=-1)

        # ---- 融合口径：原 [FUSE_MODE_PATCH] / [FUSE_TOPK_PATCH] / [FUSE_SCALE_PATCH] 逻辑原样保留 ----
        _fm = getattr(self, "fuse_mode", "topk")
        if _fm in ("sum", "prob", "mean"):
            topks = torch.arange(self.num_expert, device=prob.device).unsqueeze(0).expand(biased.size(0), -1).contiguous()
            vals = prob.gather(1, topks)
            _wbp = (_fm == "prob")
        else:
            _wbp = self.weight_by_prob
        _fk = int(getattr(self, "fuse_topk", 0) or 0)
        _is_mean = (_fm == "mean")
        if 0 < _fk <= self.num_expert:
            topks = torch.topk(biased, _fk, dim=-1)[1]
            vals = prob.gather(1, topks)
            _wbp = True
            _is_mean = False
        coeff = torch.zeros_like(prob)
        if fe is not None:
            _cval = torch.ones_like(vals)
        elif _is_mean:
            _cval = torch.full_like(vals, 1.0 / self.num_expert)
        elif _wbp:
            _cval = vals / vals.detach().sum(-1, keepdim=True)
        else:
            _cval = vals / vals.detach()
        coeff.scatter_(1, topks, _cval)
        _fs = float(getattr(self, "fuse_scale", 1.0))
        if _fs != 1.0:
            coeff = coeff * _fs

        # ---- 统计平衡 + LB：样本级计数；cnt/cnt.sum() 尺度不变（route_penalty=0 时 penalty 恒 0）----
        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()
        importance = prob.mean(dim=0)
        self.lb_loss = self.num_expert * torch.sum(importance * (cnt / cnt.sum()))

        self.update_coeff(coeff)
        self.topks = topks
        if self.record_route:
            self.route_prob_list.append(prob.detach().cpu())
'''

NEW_FORWARD = '''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE 卷积：每条样本一次专家决策，融合一次权重后做一次分组卷积。"""
        B, in_C, L = x.shape
        if self.router is not None:
            self.get_topks(x)          # 只产生一次决策：self.topks [B,K] / self.coeff [B,E]
        assert self.topks is not None and self.coeff is not None, \\
            "[SAMPLE_LEVEL] 需要路由器，或由外部注入的 coeff/topks"
        assert self.topks.dim() == 2 and self.coeff.dim() == 2, \\
            "[SAMPLE_LEVEL] 只支持样本级 2D 路由（[B,K]/[B,E]）；逐帧/逐块路由已废弃"

        final_weight, final_bias = fuse_conv_params(
            self.experts_weight, self.experts_bias, self.topks,
            self.coeff, self.weight, self.bias, self.has_bias,
            not getattr(self, "no_shared", False)
        )
        if not getattr(self, "_sample_level_logged", False):
            print("[SAMPLE_LEVEL] 样本级路由：单次融合 + 一次普通卷积", flush=True)
            self._sample_level_logged = True
        return apply_dynamic_conv1d(x, final_weight, final_bias, self.stride, self.padding,
                                    self.dilation, self.groups, self.has_bias)
'''

NEW_GET_COEFF_TOPKS = '''    def get_coeff_topks(self, embedding):
        """逐样本路由：每个样本一个专家决策，再沿时间维展开以兼容外部 hook。

        embedding: [B, T, C]；统计量 = 音频每秒的均值再对秒数求平均（与历史一致）。
        返回 coeff_frame[B,T,E] / topks_frame[B,T,K] / biased_prob_frame[B*T,E] / route_prob_frame[B*T,E]
        """
        B, T, C = embedding.shape
        F = self.frames_per_sec
        chunks = torch.split(embedding, F, dim=1)
        chunk_means = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)  # [B, nb, C]
        sample_feat = chunk_means.mean(dim=1)                                      # [B, C] <- 每样本一个统计量

        route_prob = F.softmax(self.router(sample_feat), dim=-1)     # [B, E]
        biased_prob = route_prob - self.penalty
        prob, topks = torch.topk(biased_prob, self.topk, dim=-1)     # [B, K]

        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()

        coeff = torch.zeros_like(route_prob)                        # [B, E]
        if self.weight_by_prob:
            norm_prob = prob / prob.detach().sum(-1, keepdim=True)
        else:
            norm_prob = prob / prob.detach()
        coeff.scatter_(1, topks, norm_prob)

        coeff_frame = coeff.unsqueeze(1).expand(-1, T, -1)                                  # [B, T, E]
        topks_frame = topks.unsqueeze(1).expand(-1, T, -1)                                  # [B, T, K]
        route_prob_frame = route_prob.unsqueeze(1).expand(-1, T, -1).reshape(B * T, self.num_expert)
        biased_prob_frame = biased_prob.unsqueeze(1).expand(-1, T, -1).reshape(B * T, self.num_expert)

        _lb_base = self.cnt if getattr(self, "lb_global", False) else cnt      # 兼容 [LB_GLOBAL] 全流累积口径
        importance = route_prob.mean(dim=0)
        self.lb_loss = self.num_expert * torch.sum(importance * (_lb_base / _lb_base.sum()))
        return coeff_frame, topks_frame, biased_prob_frame, route_prob_frame
'''

NEW_GET_TOPKS_BLOCK_STUB = '''    def _get_topks_block(self, x: torch.Tensor):
        """已废弃入口：直接委派给逐样本路径（保留函数名，避免外部引用报错）。"""
        return self.get_topks(x)
'''

NEW_LINEAR_GET_TOPKS = '''    def get_topks(self, x: torch.Tensor):
        """逐样本路由：整条序列一个决策。"""
        B = x.size(0)
        if x.dim() == 3:
            T = x.size(1)
            F_chunk = self.frames_per_sec
            chunks = torch.split(x, F_chunk, dim=1)
            chunk_means = torch.stack([c.mean(dim=1) for c in chunks], dim=1)  # [B, nb, In]
            flat = chunk_means.mean(dim=1)                                     # [B, In]
        else:
            T = 1
            flat = x

        prob = F.softmax(self.router(flat), dim=-1)               # [B, E]
        biased = prob - self.penalty
        vals, topks = torch.topk(biased, self.topk, dim=-1)       # [B, K]

        coeff = torch.zeros((B, self.num_expert), device=prob.device)
        if self.weight_by_prob:
            coeff.scatter_(1, topks, vals / vals.detach().sum(-1, keepdim=True))
        else:
            coeff.scatter_(1, topks, vals / vals.detach())

        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()
        self.lb_loss = self.num_expert * (prob.mean(0) * (cnt / cnt.sum())).sum()

        if x.dim() == 3:
            coeff = coeff.expand(-1, T, -1)      # [B, T, E]
            topks = topks.expand(-1, T, -1)      # [B, T, K]

        self.update_coeff(coeff)
        self.topks = topks
        if self.record_route:
            self.route_prob_list.append(prob.detach().cpu())
'''

# 按文件分组：同一文件的所有方法替换完成后，再做"文件级"静态断言与编译
FILES = [
    dict(path="src/system/moe_conv.py", forbid=["num_blocks", "unfold", "F_out_chunk"],
         require=["flat = chunk_means.mean(dim=1)"],
         edits=[dict(id="conv.get_topks", header="    def get_topks(self, x: torch.Tensor):",
                     end="    def update_batch_stats(self):", new=NEW_GET_TOPKS, tail="\n"),
                dict(id="conv.forward", header="    def forward(self, x: torch.Tensor) -> torch.Tensor:",
                     end=None, new=NEW_FORWARD, tail="")]),
    dict(path="src/system/moe_model.py", forbid=["num_blocks"],
         require=["sample_feat = chunk_means.mean(dim=1)"],
         edits=[dict(id="MoETTA.get_coeff_topks", header="    def get_coeff_topks(self, embedding):",
                     end="    def forward(self, *args, **kwargs):", new=NEW_GET_COEFF_TOPKS, tail="\n")]),
    dict(path="src/system/moe_normalization.py", forbid=["nb = (T + fps - 1) // fps"],
         require=["return self.get_topks(x)"],
         edits=[dict(id="LN._get_topks_block", header="    def _get_topks_block(self, x: torch.Tensor):",
                     end="    def update_batch_stats(self):", new=NEW_GET_TOPKS_BLOCK_STUB, tail="\n")]),
    dict(path="src/system/moe_proj.py", forbid=["num_blocks"],
         require=["flat = chunk_means.mean(dim=1)"],
         edits=[dict(id="Linear.get_topks", header="    def get_topks(self, x: torch.Tensor):",
                     end="    def step_once(self):", new=NEW_LINEAR_GET_TOPKS, tail="\n")]),
]


def do_file(f, dry):
    path = f["path"]
    if not os.path.exists(path):
        print("  [%s] 找不到文件 -> 跳过" % path)
        return False
    s = io.open(path, encoding="utf-8").read()
    n_edit = 0
    for e in f["edits"]:
        if e["header"] not in s:
            print("  [%s] FAIL 找不到方法头: %s" % (path, e["header"].strip()))
            return False
        i = s.index(e["header"])
        j = len(s) if e["end"] is None else s.index(e["end"], i + len(e["header"]))
        print("  [%s] %-24s 方法 %d 行 -> %d 行" % (os.path.basename(path), e["id"],
                                                   s[i:j].count("\n"), e["new"].count("\n") + e["tail"].count("\n")))
        s = s[:i] + e["new"] + e["tail"] + s[j:]
        n_edit += 1
    hit = [x for x in f["forbid"] if x in code_only(s)]
    if hit:
        print("  [%s] FAIL 静态断言（仅代码）：重写后仍出现 %s -> 不动该文件" % (path, hit))
        return False
    miss = [x for x in f["require"] if x not in s]
    if miss:
        print("  [%s] FAIL 静态断言：缺少 %s -> 不动该文件" % (path, miss))
        return False
    try:
        compile(s, path, "exec")
    except SyntaxError as ex:
        print("  [%s] FAIL 编译错误：%s" % (path, ex))
        return False
    print("  [%s] OK：%d 处方法重写 + 静态断言通过 + 编译通过" % (path, n_edit))
    if not dry:
        bak = path + ".bak_rw_" + time.strftime("%m%d_%H%M%S")
        shutil.copy(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        py_compile.compile(path, doraise=True)
        print("        written (backup: %s)" % bak)
    return True


def main():
    dry = "--dryrun" in sys.argv

    if not os.path.exists("_t_conv_equiv3.py"):
        print("FAIL 缺少验收工具 _t_conv_equiv3.py -> 拒绝重写（不进'改了没验'的状态）")
        sys.exit(3)

    if not dry:
        os.makedirs("/tmp/sl_ref", exist_ok=True)
        for f in FILES:
            shutil.copy(f["path"], "/tmp/sl_ref/" + os.path.basename(f["path"]).replace(".py", "_ref.py"))
        print("[ref] 重写前文件已存 -> /tmp/sl_ref/")

    ok = True
    for f in FILES:
        ok = do_file(f, dry) and ok

    if dry:
        print("\n[dryrun] 未写入任何文件")
        return
    if not ok:
        print("\nWARN 有文件未通过 -> 已成功的文件可保留；失败处请把该文件贴回来")
        sys.exit(4)

    print("\n[check] 全仓库剩余块标识（应为空）")
    for pat in ("num_blocks", "F_out_chunk", "x_unfold"):
        r = subprocess.run(["grep", "-rn", pat, "src/", "--include=*.py"], capture_output=True, text=True)
        print("  %-12s -> %s" % (pat, (r.stdout.strip().replace("\n", " | ") or "（无）")))

    print("\n[check] 等价性测试（重写前 vs 重写后，期望 max|Δ| == 0）")
    r = subprocess.run([sys.executable, "_t_conv_equiv3.py"])
    if r.returncode != 0:
        print("\nFAIL 等价性测试未通过 -> 还原：cp /tmp/sl_ref/moe_conv_ref.py src/system/moe_conv.py")
        sys.exit(5)
    print("\nPASS 重写完成 + 静态复查通过 + 逐位等价验收通过")


if __name__ == "__main__":
    main()
