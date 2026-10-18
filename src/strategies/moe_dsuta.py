# -*- coding: utf-8 -*-
"""
MoE-DSUTA：把 MoE 层放进 DSUTA 的框架（快慢双模型 + 缓冲区 + 每样本从慢模型重启）。

动机（2026-09-12 服务器实验）：
    `moe_suta`（连续适应、每样本 5 步、永不 reset）在 md_hard 上发散（WER 78.75%，
    none 43.96 / csuta 20.57 / dsuta 16.54，BA/NB 两域 WER>100%）；
    而 dsuta 同样做 13000 次适应却不发散 —— 差别就在"快慢双模型 + 缓冲区"。
    本文件用**完全相同的 MoE loss 与模型**，只替换适应机制，以分离"机制差异"与"配置差异"。

与 src/strategies/dsuta.py 的区别：
    1) 适应用 moe_suta_adapt / moe_suta_adapt_auto（MoE loss，含 EXP-K 的 lb_coef）；
    2) 快照存 CPU（省 GPU 显存；load_state_dict 会按参数设备自动搬运）。

注意：_moe_suta_adapt_loss_only 的 loss 部分是从 src/system/suta.py::moe_suta_adapt
逐行复制的（去掉 zero_grad/backward/step，用于梯度累积）。若改动 suta.py 的 loss，这里要同步。
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..system.suta import SUTASystem
from ..system.loss import softmax_entropy, mcc_loss, div_loss
from ..utils.tool import wer, batchify
from .base import IStrategy
from .dsuta import Buffer


# ------------------------------------------------------------------ MoE 适应函数
def _moe_suta_adapt_loss_only(self, wavs, record=None):
    """与 SUTASystem.moe_suta_adapt 相同的 loss，但不做 zero_grad/backward/step。"""
    if record is None:
        record = {}
    self.adapt_count += 1

    inputs = self._wav_to_model_input(wavs)
    inputs = inputs.to(device=self.model.device)

    prev_flag = getattr(self.model, "collect_stats", False)
    self.model.collect_stats = True
    try:
        self.model.current_temp = float(self.config.get("temp", 2.5))
        outputs = self.model(**inputs)
    finally:
        self.model.collect_stats = prev_flag

    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    loss = 0

    if self.config["em_coef"] > 0:
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
        x = softmax_entropy(outputs.logits / self.config["temp"])
        if self.config["non_blank"]:
            x = x[non_blank]
        if len(x) > 0:
            e_loss = x.mean(0).mean()
        else:
            e_loss = torch.tensor(0, device=self.model.device)
            record["collapse"] = True
        loss = loss + e_loss * self.config["em_coef"]
        record["e_loss"] = e_loss.item()

    if 1 - self.config["em_coef"] > 0:
        c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
        loss = loss + c_loss * (1 - self.config["em_coef"])
        record["c_loss"] = c_loss.item()

    if self.config["div_coef"] > 0:
        d_loss = div_loss(outputs.logits, self.config["non_blank"])
        loss = loss + d_loss * self.config["div_coef"]
        record["d_loss"] = d_loss.item()

    if self.config.get("l2_coef", 0.0) > 0:
        l2_loss = self.l2_loss()
        loss = loss + l2_loss * self.config.get("l2_coef", 0.0)
        record["l2_loss"] = l2_loss.item()

    # EXP-J 反 blank
    nb_ratio = (predicted_ids != 0).float().mean()
    blank_prob = outputs.logits.softmax(-1)[..., 0].mean()
    record["nb_ratio"] = nb_ratio.item()
    if nb_ratio < self.config.get("nb_threshold", 0.2):
        loss = loss + self.config.get("blank_coef", 0.1) * blank_prob
        record["blank_penalty"] = blank_prob.item()
    # [MOETTA_FILTER] 官方 MoETTA 的"动态阈值过滤 + 熵重加权"（Eq.8-10）
    #   用模型 forward 已算好的 valid_mask / current_coeff；env MOE_MOETTA_FILTER=1 才启用（默认关）
    #   实现：loss += (加权熵 − 原熵)·em_coef ⇒ 等价于把熵项替换成 Eq.9 的形式
    import os as _os
    if _os.environ.get("MOE_MOETTA_FILTER", "0") == "1" and self.config["em_coef"] > 0:
        _vm = getattr(self.model, "valid_mask", None)
        _cw = getattr(self.model, "current_coeff", None)
        if _vm is not None and _cw is not None:
            _H2 = softmax_entropy(outputs.logits / self.config["temp"])
            _m = _vm if not self.config["non_blank"] else (_vm & (predicted_ids != 0))
            _e_f = None
            if int(_m.sum()) > 0:
                _e_f = (_H2[_m] * _cw[_m]).sum() / _m.sum()
                loss = loss + (_e_f - e_loss.detach()) * self.config["em_coef"]
                record["e_loss_filtered"] = float(_e_f.item())
            self._mf_n = getattr(self, "_mf_n", 0) + 1
            _nb = int((predicted_ids != 0).sum())
            valid_n = int(_vm.sum())
            if self._mf_n % 500 == 0:
                print("[MF2] n=%d | S_t=%d | valid=%d | nonblank=%d | 帧=%d | 保留(全体)=%.1f%% 保留(nb)=%.1f%%"
                      % (self._mf_n, int(_m.sum()), valid_n, _nb, int(_m.numel()),
                         100.0*int(_m.sum())/max(1,int(_m.numel())), 100.0*int(_m.sum())/max(1,_nb)), flush=True)
                print("[MF] n=%d | 可靠帧=%d/%d (%.1f%%) | 加权熵=%s | 原熵=%.4f"
                      % (self._mf_n, int(_m.sum()), int(_m.numel()),
                         100.0 * float(_m.sum()) / max(1, int(_m.numel())),
                         ("%.4f" % float(_e_f.item())) if _e_f is not None else "n/a",
                         float(e_loss.item())), flush=True)
    # [ENTROPY_WEIGHT] Q3 软加权：低置信度样本降低该步有效学习率（不丢样本）
    #   w=(H_ema/H)^gamma，再除以 w 的 EMA ⇒ 只做再分配、不改整体步长；clamp [0.2,2.0]
    #   门控：env MOE_ENT_W=1（默认关 ⇒ 与历史完全一致）；gamma = config.ent_w_gamma（默认 0.5）
    import os
    if os.environ.get("MOE_ENT_W", "0") == "1":
        try:
            _H = float(e_loss.detach().item())
        except Exception:
            _H = 0.0
        if getattr(self, "_ent_H_ema", None) is None:
            self._ent_H_ema, self._ent_w_ema = max(_H, 1e-6), 1.0
        _gam = float(self.config.get("ent_w_gamma", 0.5))
        _mode = os.environ.get("MOE_ENTW_MODE", "soft").strip().lower()
        if _mode == "hard":
            _w = 0.0 if _H > float(self.config.get("ent_w_hard_k", 1.5)) * self._ent_H_ema else 1.0
        else:
            _w = (self._ent_H_ema / max(_H, 1e-6)) ** _gam
            _w = min(max(_w / max(self._ent_w_ema, 1e-6), 0.2), 2.0)
        loss = loss * _w
        self._ent_n = getattr(self, "_ent_n", 0) + 1
        if _w < 0.5:
            self._ent_drop = getattr(self, "_ent_drop", 0) + 1
        if self._ent_n % 500 == 0:
            print("[ENT_W] n=%d | 丢弃步数(w<0.5)=%d (%.1f%%) | H=%.4f H_ema=%.4f | mode=%s"
                  % (self._ent_n, getattr(self, "_ent_drop", 0),
                     100.0 * getattr(self, "_ent_drop", 0) / self._ent_n, _H, self._ent_H_ema, _mode), flush=True)
        self._ent_H_ema = 0.99 * self._ent_H_ema + 0.01 * _H
        self._ent_w_ema = 0.99 * self._ent_w_ema + 0.01 * _w
        record["ent_w"] = _w
        if not getattr(self, "_ent_w_logged", False):
            print("[ENT_W] 软加权启用 | gamma=%.2f | 首样本 H=%.4f w=%.4f" % (_gam, _H, _w), flush=True)
            self._ent_w_logged = True

    # EXP-K：官方 MoETTA 的 LB 项（给 router 一阶学习信号）
    lb_coef = float(self.config.get("lb_coef", 0.0))
    moe_lb = getattr(self.model, "moe_lb_loss", None)
    if lb_coef > 0 and moe_lb is not None:
        loss = loss + lb_coef * moe_lb
        record["lb_loss"] = float(moe_lb.item())
        record["lb_term"] = lb_coef * float(moe_lb.item())

    record["total_loss"] = loss.item()
    return loss


def _moe_suta_adapt_auto(self, wavs, batch_size=-1, record=None):
    """DSUTA 慢模型用的批量更新：梯度累积后只 step 一次。"""
    if record is None:
        record = {}
    self.adapt_count += 1
    # [SLOW_LR] 只覆盖本实例(=慢系统)optimizer 的 lr；快路是另一个实例，不受影响
    import os as _os_slr
    _slr = _os_slr.environ.get("MOE_SLOW_LR", "").strip()
    if _slr:
        _slr_v = float(_slr)
        for _slr_g in self.optimizer.param_groups:
            _slr_g["lr"] = _slr_v
        if not getattr(self, "_slow_lr_logged", False):
            print("[SLOW_LR] 慢更新 lr 覆盖为 %.3e（%d 个参数组）"
                  % (_slr_v, len(self.optimizer.param_groups)), flush=True)
            self._slow_lr_logged = True
    if batch_size == -1:
        batch_size = len(wavs)
    self.model.zero_grad()
    denom_scale = len(wavs) // batch_size
    assert denom_scale > 0
    for chunk in batchify(wavs, batch_size=batch_size):
        loss = self.moe_suta_adapt_loss_only(chunk, record=record)
        self.adapt_count -= 1          # 避免重复计数
        (loss / denom_scale).backward()
    self.optimizer.step()
    if self.scheduler is not None:
        self.scheduler.step()
    self.model.zero_grad()


SUTASystem.moe_suta_adapt_loss_only = _moe_suta_adapt_loss_only
SUTASystem.moe_suta_adapt_auto = _moe_suta_adapt_auto


# ------------------------------------------------------------------ CPU 快照
def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def snapshot_cpu(system, key):
    """把 (模型状态, 优化器状态) 存到 CPU；替换掉 SUTASystem 默认的 GPU deepcopy 快照。"""
    opt_state = None
    if getattr(system, "optimizer", None) is not None:
        opt_state = _to_cpu(system.optimizer.state_dict())
    system.history[key] = (_to_cpu(system.model.state_dict()), opt_state, None)


def load_snapshot_cpu(system, key):
    model_state, opt_state, _ = system.history[key]
    system.model.load_state_dict(model_state, strict=True)
    if opt_state is not None and getattr(system, "optimizer", None) is not None:
        # [FS_SLOW_PATCH] 只接收 Adam state; PyTorch load_state_dict 会用保存方的
        #   param_groups 超参覆盖接收方 => 会把慢模型 shared 组 lr=0 复制到快模型。
        #   对历史配置无损(两 system 超参本来就相同 => 覆盖与否等价)。
        _keep = [{k: v for k, v in g.items() if k not in ("params", "param_names")}
                 for g in system.optimizer.param_groups]
        system.optimizer.load_state_dict(opt_state)
        for _g, _k in zip(system.optimizer.param_groups, _keep):
            _g.update(_k)


# ------------------------------------------------------------------ 策略
class MOE_DSUTAStrategy(IStrategy):
    def _mask_fast_top1(self, system):
        """[FAST_MASK_PATCH] 快路径只保留 共享(基座) + 该层 top-1 专家 的更新。
        路由: p.grad=None => step 跳过、状态不动。
        其余专家: grad 切片清零 + Adam 状态切片清零(否则沿旧动量滑行)。
        返回前 10 次调用的 (param, sel, before) 供自检。"""
        import torch as _tt
        info = []
        for _m in getattr(system.model, "moe_layer", []):
            _r = getattr(_m, "router", None)
            if _r is not None:
                for _p in _r.parameters():
                    _p.grad = None
            for _n in ("experts_weight", "experts_bias"):
                _p = getattr(_m, _n, None)
                if not isinstance(_p, _tt.nn.Parameter) or _p.grad is None:
                    continue
                _E = int(_p.shape[0])
                if _E <= 1:
                    continue
                _sel = None
                _tk = getattr(_m, "topks", None)
                if _tk is not None:
                    try:
                        _sel = int(_tk.reshape(-1)[0].item())
                    except Exception:
                        _sel = None
                if _sel is None or not (0 <= _sel < _E):
                    _sel = int(_p.grad.reshape(_E, -1).norm(dim=1).argmax().item())
                for _e in range(_E):
                    if _e != _sel:
                        _p.grad[_e] = 0
                _st = system.optimizer.state.get(_p, None)
                if _st:
                    for _k in ("exp_avg", "exp_avg_sq"):
                        _v = _st.get(_k, None)
                        if _v is not None and _v.shape[0] == _E:
                            for _e in range(_E):
                                if _e != _sel:
                                    _v[_e] = 0
                if len(info) < 10:
                    info.append((_p, _sel, _p.data.detach().clone()))
        return info

    def _fast_mask_step(self, *a, **kw):
        """[FAST_MASK_PATCH] 包在快模型 optimizer.step 外面。"""
        import torch as _tt
        info = []
        try:
            info = self._mask_fast_top1(self.system)
        except Exception as _e:
            print("[FAST_MASK] mask failed: %s" % _e, flush=True)
        _ret = self._orig_step(*a, **kw)
        try:
            for _p, _sel, _before in info:
                _d = (_p.data - _before).abs()
                _E = int(_p.shape[0])
                _msk = _tt.ones(_E, dtype=_tt.bool, device=_d.device)
                _msk[_sel] = False
                _ex = float(_d[_msk].max()) if bool(_msk.any()) else 0.0
                _se = float(_d[_sel].max())
                self._fm_excl = max(getattr(self, "_fm_excl", 0.0), _ex)
                self._fm_sel = max(getattr(self, "_fm_sel", 0.0), _se)
        except Exception as _e:
            print("[FAST_MASK] check failed: %s" % _e, flush=True)
        return _ret

    def _shared_vec(self, system):
        """[FS_SLOW_PATCH] MoE 层共享(基座) weight/bias 拼成 CPU float32 向量。"""
        import torch as _tt
        vs = []
        for _m in getattr(system.model, "moe_layer", []):
            for _n in ("weight", "bias"):
                _p = getattr(_m, _n, None)
                if isinstance(_p, _tt.nn.Parameter):
                    vs.append(_p.data.detach().float().cpu().reshape(-1))
        return _tt.cat(vs) if vs else None

    def _regroup_shared(self, system, shared_lr):
        """[FS_SLOW_PATCH] 共享(基座)权重单独成一组; shared_lr=0.0 即冻结该组
        (AdamW 下 lr=0 连 weight decay 一起失效)。两个 system 都调用 => 分组同构,
        保证 load_snapshot_cpu 里的 optimizer.load_state_dict 仍可用。"""
        import os as _os_fs
        opt = getattr(system, "optimizer", None)
        if opt is None:
            return
        sh_ids, ex_ids = set(), set()
        for _m in getattr(system.model, "moe_layer", []):
            for _n in ("weight", "bias"):
                _t = getattr(_m, _n, None)
                if isinstance(_t, torch.nn.Parameter):
                    sh_ids.add(id(_t))
            for _n in ("experts_weight", "experts_bias"):
                _t = getattr(_m, _n, None)
                if isinstance(_t, torch.nn.Parameter):
                    ex_ids.add(id(_t))
            _r = getattr(_m, "router", None)
            if _r is not None:
                for _p in _r.parameters():
                    ex_ids.add(id(_p))
        all_p = [p for g in opt.param_groups for p in g["params"]]
        g_sh = [p for p in all_p if id(p) in sh_ids]
        g_ex = [p for p in all_p if id(p) in ex_ids and id(p) not in sh_ids]
        g_rs = [p for p in all_p if id(p) not in sh_ids and id(p) not in ex_ids]
        tpl = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
        _elr = _os_fs.environ.get("MOE_EXPERT_LR", "").strip()
        groups = []
        if g_sh:
            _d = dict(tpl); _d["params"] = g_sh
            if shared_lr is not None:
                _d["lr"] = float(shared_lr)
            groups.append(_d)
        if g_ex:
            _d = dict(tpl); _d["params"] = g_ex
            if _elr:
                _d["lr"] = float(_elr)
            groups.append(_d)
        if g_rs:
            _d = dict(tpl); _d["params"] = g_rs
            groups.append(_d)
        opt.param_groups = groups
        print("[FS_SLOW] shared_lr=%s | shared=%d 张量/%.3fM | experts=%d 张量 | rest=%d 张量"
              % (shared_lr, len(g_sh), sum(p.numel() for p in g_sh) / 1e6,
                 len(g_ex), len(g_rs)), flush=True)

    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        self.slow_system = SUTASystem(config["system_config"])
        self.slow_system.eval()

        # [FS_SLOW_PATCH] MOE_FREEZE_SHARED=slow: 只在【慢模型】冻结共享(基座)权重
        import os as _os_fs
        if _os_fs.environ.get("MOE_FREEZE_SHARED", "").strip().lower() == "slow":
            # [ES_LR_SCOPE] MOE_EXPERT_LR 默认只作用于慢模型（MOE_EXPERT_LR_SCOPE=both 恢复旧行为）
            _esc = _os_fs.environ.get("MOE_EXPERT_LR_SCOPE", "slow").strip().lower()
            _elr_keep = _os_fs.environ.pop("MOE_EXPERT_LR", None)
            if _esc == "both" and _elr_keep is not None:
                _os_fs.environ["MOE_EXPERT_LR"] = _elr_keep
            self._regroup_shared(self.system, None)      # 快模型: shared 组继承基准 lr
            if _elr_keep is not None:
                _os_fs.environ["MOE_EXPERT_LR"] = _elr_keep
            self._regroup_shared(self.slow_system, 0.0)  # 慢模型: shared 组 lr=0 -> 真冻结
            # [ES_LR_SCOPE] 自检：两个 system 每组的 lr 与张量数
            for _nm, _sy in (("fast", self.system), ("slow", self.slow_system)):
                try:
                    _nmap = {id(_p): _n for _n, _p in _sy.model.named_parameters()}
                    _gl = " | ".join(
                        "lr=%s x%d {%s}" % (g.get("lr"), len(g["params"]),
                                            ", ".join([_nmap.get(id(_p), "?") for _p in g["params"]][:3]))
                        for g in _sy.optimizer.param_groups)
                    print("[ES_LR_SCOPE] %s groups: %s" % (_nm, _gl), flush=True)
                except Exception as _e:
                    print("[ES_LR_SCOPE] %s print failed: %s" % (_nm, _e), flush=True)
            self._fs_ref = {"slow": self._shared_vec(self.slow_system),
                            "fast": self._shared_vec(self.system)}

        # 用 CPU 快照覆盖默认的 GPU 快照（释放显存）
        for sys_ in (self.system, self.slow_system):
            snapshot_cpu(sys_, "init")
            snapshot_cpu(sys_, "start")

        # [FUSE_PER_MODEL] MOE_FAST_FUSE 只改 fast 的 fuse_mode（slow 保持全局值）
        import os as _os_fpm
        _fpm = _os_fpm.environ.get("MOE_FAST_FUSE", "").strip().lower()
        if _fpm:
            for _m in self.system.model.moe_layer:
                _m.fuse_mode = _fpm
            print("[FUSE_PER_MODEL] fast=" + _fpm + " | slow=" + str(self.slow_system.model.moe_layer[0].fuse_mode), flush=True)
        # [FAST_MASK_PATCH] MOE_FAST_TOPK1=1: 快路径只更新 共享 + 该层 top-1 专家
        import os as _os_fm
        if _os_fm.environ.get("MOE_FAST_TOPK1", "0").strip() == "1":
            self._orig_step = self.system.optimizer.step
            self.system.optimizer.step = self._fast_mask_step
            self._fm_excl, self._fm_sel = 0.0, 0.0
            print("[FAST_MASK] enable: fast path updates SHARED + top-1 expert only (router/other experts frozen)", flush=True)

        self.timestep = 0
        self.update_freq = self.strategy_config.get("update_freq", 5)
        self.memory = Buffer(max_size=self.strategy_config.get("memory", 5))

        # [DYN_RESET] 官方 DynamicReset（loss 偏差检测器）移植：门控 MOE_DYN_RESET=1
        import os as _os_dyn
        self._dyn_on = _os_dyn.environ.get("MOE_DYN_RESET", "0").strip() == "1"
        self._dyn_K = int(_os_dyn.environ.get("MOE_DYN_K", "100"))
        self._dyn_pat = int(_os_dyn.environ.get("MOE_DYN_PATIENCE", "2"))
        self._dyn_dev = float(_os_dyn.environ.get("MOE_DYN_DEV", "2.0"))
        self._dyn_dwell = int(_os_dyn.environ.get("MOE_DYN_DWELL", "0"))
        import os as _os_lk
        self._leak_beta = float(_os_lk.environ.get("MOE_LEAK_BETA", "0") or "0")
        self._leak_scope = _os_lk.environ.get("MOE_LEAK_SCOPE", "all").strip().lower()
        self._leak_trace = _os_lk.environ.get("MOE_LEAK_TRACE", "0").strip() == "1"
        self._leak_cache = {}
        if self._leak_beta > 0:
            print("[LEAK_BETA] enable: beta=%.4f scope=%s trace=%d"
                  % (self._leak_beta, self._leak_scope, 1 if self._leak_trace else 0), flush=True)
        self._dyn_mu = None
        self._dyn_sd = None
        self._dyn_stats = []
        self._dyn_last = 0
        self._dyn_fail = 0
        self._dyn_n = 0
        if self._dyn_on:
            print("[DYN_RESET] enable: K=%d patience=%d dev=%.2f" % (self._dyn_K, self._dyn_pat, self._dyn_dev), flush=True)
            import os as _os6
            print("[DYN_STAGE1] win=%s persist=%s dev_lo=%s dev=%s dwell=%s (win unset/1 = legacy detector)"
                  % (_os6.environ.get("MOE_DYN_WIN", "unset"), _os6.environ.get("MOE_DYN_PERSIST", "unset"),
                     _os6.environ.get("MOE_DYN_DEV_LO", "unset"), _os6.environ.get("MOE_DYN_DEV", "unset"),
                     _os6.environ.get("MOE_DYN_DWELL", "unset")), flush=True)

    # ==================== [DYN_RESET] 官方 DynamicReset 移植 ====================
    def _dyn_losses(self, wavs, snap):
        """逐样本返回 [loss(在 snap 快照下) - loss(在 init 快照下)]。"""
        load_snapshot_cpu(self.slow_system, "init")
        self.slow_system.eval()
        li = [self.slow_system.calc_suta_loss([w])["total_loss"] for w in wavs]
        load_snapshot_cpu(self.slow_system, snap)
        self.slow_system.eval()
        ls = [self.slow_system.calc_suta_loss([w])["total_loss"] for w in wavs]
        return [float(x) - float(y) for x, y in zip(ls, li)]

    def _dyn_build(self, sample):
        """标定：K//2 处快照 domain，收集 K//2..K 的改善量，在 K 处拟合高斯。"""
        if not self._dyn_on:
            return
        t = self.timestep + 1
        if self._dyn_mu is not None:
            return
        if t == self._dyn_last + self._dyn_K // 2:
            snapshot_cpu(self.slow_system, "domain")
        elif self._dyn_last + self._dyn_K // 2 <= t < self._dyn_last + self._dyn_K:
            self._dyn_stats.extend(self._dyn_losses([sample["wav"]], "domain"))
        if t == self._dyn_last + self._dyn_K and len(self._dyn_stats) >= 2:
            import statistics as _st
            self._dyn_mu = _st.mean(self._dyn_stats)
            self._dyn_sd = _st.stdev(self._dyn_stats)
            self._dyn_stats = []
            print("[DYN_RESET] calib done at t=%d: mu=%.4f sd=%.4f" % (t, self._dyn_mu, self._dyn_sd), flush=True)

    def _dyn_should_reset(self, wavs):
        """[DYN_STAGE1] windowed mean of dev + persistence + dwell. MOE_DYN_WIN<=1 => legacy."""
        if not self._dyn_on or self._dyn_mu is None or not wavs:
            return False, 0.0
        import os as _os1
        import statistics as _st1

        x = _st1.mean(self._dyn_losses(wavs, "domain"))
        den = self._dyn_sd / (len(wavs) ** 0.5)
        dev = 0.0 if den <= 0 else (x - self._dyn_mu) / den

        _win = int(_os1.environ.get("MOE_DYN_WIN", "1") or "1")
        _per = int(_os1.environ.get("MOE_DYN_PERSIST", "0") or "0")
        _lo = float(_os1.environ.get("MOE_DYN_DEV_LO", "0") or "0")
        _trc = _os1.environ.get("MOE_DYN_TRACE", "0").strip() == "1"

        _gap = self.timestep + 1 - self._dyn_last
        hist = getattr(self, "_dyn_hist", None)
        if hist is None:
            hist = self._dyn_hist = []
        hist.append(dev)
        if len(hist) > max(1, _win):
            del hist[:len(hist) - max(1, _win)]

        if _win > 1:
            _dbar = _st1.mean(hist)
            _cnt = sum(1 for _d in hist if _d > _lo)
            _full = len(hist) >= _win
            _ok = bool(_full and (_dbar > self._dyn_dev) and (_cnt >= _per) and (_gap >= self._dyn_dwell))
        else:
            _dbar = dev
            _cnt = 1 if dev > _lo else 0
            _full = True
            if dev > self._dyn_dev:
                self._dyn_fail += 1
            else:
                self._dyn_fail = 0
            _ok = bool((self._dyn_fail == self._dyn_pat) and (_gap >= self._dyn_dwell))

        if _trc:
            print("[DYN_TRACE] t=%d x=%.4f dev=%.3f dbar=%.3f cnt=%d/%d gap=%d full=%d win=%d hit=%d"
                  % (self.timestep + 1, x, dev, _dbar, _cnt, len(hist), _gap,
                     1 if _full else 0, _win, 1 if _ok else 0), flush=True)

        if _ok:
            self._dyn_hist = []
        return _ok, dev

    def _dyn_partial_sets(self, sys_):
        """[DYN_PARTIAL] 返回 (shared_keys, shared_params, expert_keys, expert_params, norm_keys, norm_params)。"""
        import os as _osp  # noqa: F401
        shared_k, shared_p, expert_k, expert_p = [], [], [], []
        moes = {id(m) for m in (getattr(sys_.model, "moe_layer", None) or [])}
        for nm, mod in sys_.model.named_modules():
            if id(mod) not in moes:
                continue
            for pn in ("weight", "bias"):
                _p = getattr(mod, pn, None)
                if isinstance(_p, torch.nn.Parameter):
                    shared_k.append(nm + "." + pn)
                    shared_p.append(_p)
            for pn in ("experts_weight", "experts_bias"):
                _p = getattr(mod, pn, None)
                if isinstance(_p, torch.nn.Parameter):
                    expert_k.append(nm + "." + pn)
                    expert_p.append(_p)
            _r = getattr(mod, "router", None)
            if _r is not None:
                for _rn, _p in _r.named_parameters():
                    expert_k.append(nm + ".router." + _rn)
                    expert_p.append(_p)
        norm_ids = {id(m) for m in (getattr(sys_.model, "normal_layers", None) or [])}
        norm_k, norm_p = [], []
        for nm, mod in sys_.model.named_modules():
            if id(mod) in norm_ids:
                for _pn, _p in mod.named_parameters():
                    if _pn in ("weight", "bias"):
                        norm_k.append(nm + "." + _pn)
                        norm_p.append(_p)
        return shared_k, shared_p, expert_k, expert_p, norm_k, norm_p

    def _dyn_partial_reset(self, mode):
        """[DYN_PARTIAL] 只回滚选中子集；返回 True=成功，False=请回退全量重置。"""
        import os as _osp
        sys_ = self.slow_system
        sk, sp, ek, ep, nk, npp = self._dyn_partial_sets(sys_)
        if mode == "shared":
            keys, params = sk, sp
        elif mode == "experts":
            keys, params = ek, ep
        elif mode == "shared_norm":
            keys, params = sk + nk, sp + npp
        else:
            print("[DYN_PARTIAL] 未知 MOE_DYN_PARTIAL=%r ⇒ 回退全量重置" % mode, flush=True)
            return False
        init_sd = sys_.history["init"][0]
        sel = {}
        miss = []
        for k in keys:
            if k in init_sd:
                sel[k] = init_sd[k]
            else:
                miss.append(k)
        if not sel:
            print("[DYN_PARTIAL] mode=%s 一个 key 都没匹配上（shared=%d experts=%d norm=%d）⇒ 回退全量重置"
                  % (mode, len(sk), len(ek), len(nk)), flush=True)
            return False
        if miss:
            print("[DYN_PARTIAL] mode=%s 有 %d/%d 个构造出的 key 不在 init 快照里：%s"
                  % (mode, len(miss), len(keys), miss[:5]), flush=True)
        ret = sys_.model.load_state_dict(sel, strict=False)
        _ue = getattr(ret, "unexpected_keys", ["MISSING_ATTR"])
        nz = 0
        if _osp.environ.get("MOE_DYN_PARTIAL_OPT", "1").strip() == "1":
            for _p in params:
                if sys_.optimizer.state.pop(_p, None) is not None:
                    nz += 1
        print("[DYN_PARTIAL] mode=%s 回滚 %d 个张量（shared=%d experts=%d norm=%d，选中=%d）| "
              "unexpected=%d | 清 Adam 动量=%d"
              % (mode, len(sel), len(sk), len(ek), len(nk), len(params), len(_ue), nz), flush=True)
        if _ue:
            print("[DYN_PARTIAL] %d 个 key 不在模型里：%s"
                  % (len(_ue), list(_ue)[:5]), flush=True)
        return True

    def _dyn_do_reset(self, dev):
        """重置：全量（默认）或部分（MOE_DYN_PARTIAL）-> 重拍 start -> 检测器状态清零。"""
        import os as _osp
        self._dyn_n += 1
        _pm = _osp.environ.get("MOE_DYN_PARTIAL", "").strip()
        _done = False
        if _pm:
            try:
                _done = self._dyn_partial_reset(_pm)
            except Exception as _e:
                print("[DYN_PARTIAL] 失败（%s）⇒ 回退全量重置" % _e, flush=True)
                _done = False
        if not _done:
            load_snapshot_cpu(self.slow_system, "init")
        snapshot_cpu(self.slow_system, "start")
        self.system.history["start"] = self.slow_system.history["start"]
        self._dyn_mu = None
        self._dyn_sd = None
        self._dyn_stats = []
        self._dyn_last = self.timestep + 1
        self._dyn_fail = 0
        self._dyn_hist = []
        self._dyn_last_reset_t = self.timestep + 1
        print("[DYN_RESET] RESET #%d at timestep=%d (dev=%.2f, partial=%s)"
              % (self._dyn_n, self.timestep, dev, _pm if _done else "none"), flush=True)

        # [SLOWCOMMIT] 重置后开窗（MOE_SB_WIN=0 => 不开）
        if getattr(self, "_sb_win", 0) > 0:
            self._sb_until = self.timestep + self._sb_win
            self._sb_since = 0
            print("[SLOWCOMMIT] OPEN t=%d -> %d (uf=%d slide=%d)"
                  % (self.timestep, self._sb_until, self._sb_uf, self._sb_slide), flush=True)

        # [WARM_PATCH] open warm-up window (MOE_WARM_WIN=0 => off)
        _ww = int(_osp.environ.get("MOE_WARM_WIN", "0") or 0)
        if _ww > 0:
            self._warm_mult = int(_osp.environ.get("MOE_WARM_MULT", "2") or 2)
            self._warm_until = self.timestep + _ww
            self._warm_n = 0
            self._warm_k = 0
            print("[WARM] OPEN t=%d -> %d (fast steps x%d)"
                  % (self.timestep, self._warm_until, self._warm_mult), flush=True)

    def _leak_to_init(self, sys_, beta):
        """[LEAK_BETA] theta <- (1-beta)*theta + beta*theta_init on the slow system (no grad)."""
        init_sd = sys_.history["init"][0]
        scope = getattr(self, "_leak_scope", "all")
        if scope == "all":
            params = [(n, p) for n, p in sys_.model.named_parameters() if p.requires_grad]
        else:
            sk, sp, ek, ep, nk, npp = self._dyn_partial_sets(sys_)
            if scope == "shared":
                params = list(zip(sk, sp))
            elif scope == "experts":
                params = list(zip(ek, ep))
            elif scope == "shared_norm":
                params = list(zip(sk + nk, sp + npp))
            else:
                params = [(n, p) for n, p in sys_.model.named_parameters() if p.requires_grad]
        cache = getattr(self, "_leak_cache", None)
        if cache is None:
            cache = self._leak_cache = {}
        n_ok = 0
        n_miss = 0
        for name, p in params:
            ref = cache.get(name)
            if ref is None or ref.device != p.device:
                cand = init_sd.get(name)
                if cand is None:
                    n_miss += 1
                    continue
                ref = cand.to(p.device)
                cache[name] = ref
            p.data.mul_(1.0 - beta).add_(ref, alpha=beta)
            n_ok += 1
        return n_ok, n_miss

    def _init_start(self, sample):
        load_snapshot_cpu(self.system, "start")

    def _adapt(self, sample):
        self.system.eval()
                # [WARM_PATCH] post-shift warm-up: within W samples after a reset, fast steps x M
        _kw = int(getattr(self, "_warm_mult", 0) or 0)
        _nsteps = self.strategy_config["steps"]
        if _kw > 1 and self.timestep < getattr(self, "_warm_until", 0):
            _nsteps = _nsteps * _kw
            self._warm_n = getattr(self, "_warm_n", 0) + 1
            self._warm_k = getattr(self, "_warm_k", 0) + _kw
            if self._warm_n % 20 == 0:
                print("[WARM] in-window #%d (t=%d, steps=%d, until=%d, extra=%d)"
                      % (self._warm_n, self.timestep, _nsteps, self._warm_until, self._warm_k), flush=True)
        for _ in range(_nsteps):
            self.system.moe_suta_adapt(wavs=[sample["wav"]], record={})
        # [BR] 该样本适应结束：把本样本统计量并入 EMA，供下一条样本做 batch 级专家决策
        if getattr(self.system.model, "batch_route_cfg", False):
            self.system.model.update_batch_stats()

    def _update(self, sample):
        self.memory.update(sample)
        # [DYN_RESET] 每 update_freq 条检测一次（作用于 buffer）；命中则本步"只重置不更新"
        if self._dyn_on and (self.timestep + 1) % self.update_freq == 0:
            try:
                _dyres, _dydev = self._dyn_should_reset([s["wav"] for s in self.memory.data])
                self._last_dev = _dydev
            except Exception as _e:
                _dyres, _dydev = False, 0.0
                print("[DYN_RESET] detect failed: %s" % _e, flush=True)
            if _dyres:
                self._dyn_do_reset(_dydev)
                self.memory.clear()
                torch.cuda.empty_cache()
                self.system.history["start"] = self.slow_system.history["start"]
                return
        if getattr(self, "_sb_win", 0) > 0:
            # [SB_ADAPT] open/extend the dense phase when drift is elevated
            if getattr(self, "_sb_adapt", False) and getattr(self, "_last_dev", None) is not None:
                if self._last_dev >= self._sb_dev_hi:
                    _sb_was_closed = self.timestep >= self._sb_until
                    _sb_until_new = self.timestep + (self._sb_adapt_win or self._sb_win)
                    if _sb_until_new > self._sb_until:
                        self._sb_until = _sb_until_new
                        if _sb_was_closed:
                            self._sb_ad_n += 1
                            print("[SB_ADAPT] open dense window #%d t=%d dev=%.2f until=%d"
                                  % (self._sb_ad_n, self.timestep, self._last_dev, self._sb_until), flush=True)
            _sb_k = self._sb_uf if self.timestep < self._sb_until else self._sb_after
            self._sb_since += 1
            _sb_need = self.memory.max_size if self._sb_slide else _sb_k
            _sb_fire = (self._sb_since >= _sb_k) and (len(self.memory.data) >= min(_sb_need, self.memory.max_size))
        else:
            _sb_fire = ((self.timestep + 1) % self.update_freq == 0)
        # [SEG] fire by total audio seconds in buffer (stateless: safe across clears/resets)
        if getattr(self, "_seg_sec", 0) > 0:
            _seg_dur = sum(len(_s["wav"]) for _s in self.memory.data) / 16000.0
            _sb_fire = (_seg_dur >= self._seg_sec)
        if _sb_fire:
            if getattr(self, "_sb_win", 0) > 0:
                self._sb_since = 0
                if self.timestep < self._sb_until:
                    self._sb_n += 1
                    if self._sb_n % 10 == 0:
                        print("[SLOWCOMMIT] in-window commit #%d t=%d batch=%d k=%d"
                              % (self._sb_n, self.timestep, len(self.memory.data), _sb_k), flush=True)
            if getattr(self, "_sbatch", 1) != 1:
                self._sb_c = getattr(self, "_sb_c", 0) + 1
                if self._sb_c % 100 == 0:
                    _sb_b = len(self.memory.data)
                    print("[SLOWCOMMIT] batch #%d t=%d buffer=%d batch=%d"
                          % (self._sb_c, self.timestep, _sb_b,
                             _sb_b if self._sbatch == -1 else min(self._sbatch, _sb_b)), flush=True)
            load_snapshot_cpu(self.slow_system, "start")
            self.slow_system.eval()
            record = {}
            _seg_wavs = [s["wav"] for s in self.memory.data]
            if getattr(self, "_seg_sec", 0) > 0 and _seg_wavs:
                import numpy as _np
                # concatenate as ARRAYS, never as list(wav): the processor rejects
                # a list of numpy.float32 scalars (that was the v1 crash)
                _seg_all = _np.concatenate([_np.asarray(_w).reshape(-1) for _w in _seg_wavs])
                _seg_len = max(1, int(getattr(self, "_seg_chunk", 3.0) * 16000))
                _seg_num = len(_seg_all) // _seg_len
                _seg_wavs = [_seg_all[_i * _seg_len:(_i + 1) * _seg_len] for _i in range(_seg_num)]
                self._seg_n = getattr(self, "_seg_n", 0) + 1
                _seg_avg = (sum(len(_w) for _w in _seg_wavs) / max(1, len(_seg_wavs)) / 16000.0)
                print("[SEG] commit #%d: buf=%.1fs -> %d chunks (avg %.2fs, type %s)"
                      % (self._seg_n, len(_seg_all) / 16000.0, len(_seg_wavs), _seg_avg,
                         type(_seg_wavs[0]).__name__ if _seg_wavs else "-"), flush=True)
                if _seg_num == 0:
                    print("[SEG] WARN: buffer shorter than one chunk, commit skipped", flush=True)
            if _seg_wavs:
                self.slow_system.moe_suta_adapt_auto(
                    wavs=_seg_wavs,
                batch_size=getattr(self, "_sbatch", 1),
                record=record,
            )
            _lnk = getattr(self, "_leak_beta", 0.0)
            if _lnk > 0:
                try:
                    _n_ok, _n_miss = self._leak_to_init(self.slow_system, _lnk)
                    if getattr(self, "_leak_trace", False) or _n_miss:
                        print("[LEAK_BETA] t=%d leaked=%d miss=%d beta=%.4f scope=%s"
                              % (self.timestep + 1, _n_ok, _n_miss, _lnk,
                                 getattr(self, "_leak_scope", "all")), flush=True)
                except Exception as _e:
                    print("[LEAK_BETA] failed: %s" % _e, flush=True)
            snapshot_cpu(self.slow_system, "start")
            if not (getattr(self, "_sb_win", 0) > 0 and getattr(self, "_sb_slide", 0)):
                self.memory.clear()
            torch.cuda.empty_cache()
        # 快模型下一次从这里重启（指向慢模型的最新提交状态）
        self.system.history["start"] = self.slow_system.history["start"]

    def run(self, ds: Dataset):
        long_cnt = 0
        n_words, errs, losses, transcriptions = [], [], [], []
        # [BR] 流开始：清空 batch 级路由统计量；慢模型（锚点）保持逐样本路由
        if getattr(self.system.model, "batch_route_cfg", False):
            self.system.model.reset_batch_stats()
            self.slow_system.model.set_batch_route(False)
        # [ORACLE] 专家=域 的上限实验：按样本 tid 强制专家号（None = 正常路由）
        _moe_cfg = (self.config.get("system_config", {}) or {}).get("moetta", {}) or {}
        oracle = bool(_moe_cfg.get("oracle_route", False))
        _orset = bool(_moe_cfg.get("oracle_reset", False))
        _bnd = set(getattr(ds, "task_boundaries", []) or [])
        print("[ORACLE_RESET] enable=%s | boundaries=%s" % (_orset, sorted(_bnd)), flush=True)
        self._rcand = 0
        self._rlast = 0
        print("[ORACLE_RESET] gates: min_dwell=%s every_k=%s" % (_moe_cfg.get("reset_min_dwell", 0), _moe_cfg.get("reset_every_k", 1)), flush=True)
        # [SLOWCOMMIT] 读 env（默认全关 => 行为与历史逐字一致）
        import os as _os_sb
        self._sb_win = int(_os_sb.environ.get("MOE_SB_WIN", "0") or 0)
        self._sb_uf = int(_os_sb.environ.get("MOE_SB_UF", "2") or 2)
        self._sb_after = int(_os_sb.environ.get("MOE_SB_AFTER", "5") or 5)
        self._sb_slide = int(_os_sb.environ.get("MOE_SB_SLIDE", "0") or 0)
        self._sbatch = int(_os_sb.environ.get("MOE_SBATCH", "1") or 1)
        self._sb_adapt = int(_os_sb.environ.get("MOE_SB_ADAPT", "0") or 0) > 0
        self._sb_dev_hi = float(_os_sb.environ.get("MOE_SB_DEV_HI", "1.0") or 1.0)
        self._sb_adapt_win = int(_os_sb.environ.get("MOE_SB_ADAPT_WIN", "0") or 0)
        self._sb_ad_n = 0
        self._last_dev = None
        self._sb_until = 0
        self._sb_since = 0
        self._sb_n = 0
        self._sb_c = 0
        print("[SLOWCOMMIT] enable win=%d uf=%d after=%d slide=%d sbatch=%d"
              % (self._sb_win, self._sb_uf, self._sb_after, self._sb_slide, self._sbatch), flush=True)
        # [SEG] duration-based buffer + fixed chunks (off by default)
        import os as _os_seg
        self._seg_sec = float(_os_seg.environ.get("MOE_SEG_SEC", "0") or 0)
        self._seg_chunk = float(_os_seg.environ.get("MOE_SEG_CHUNK", "3.0") or 3.0)
        self._seg_n = 0
        if self._seg_sec > 0:
            print("[SEG] enable thresh=%.1fs chunk=%.1fs (buffer max_size=%s samples)"
                  % (self._seg_sec, self._seg_chunk, getattr(self.memory, "max_size", "?")), flush=True)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            if oracle:
                tid = int(sample.get("tid", -1))
                n_exp = self.system.model.moe_layer[0].num_expert
                self.system.model.set_forced_expert(tid % n_exp if tid >= 0 else None)

            if _orset and self.timestep in _bnd:
                self._rcand += 1
                _dwell = int(_moe_cfg.get("reset_min_dwell", 0) or 0)
                _everyk = int(_moe_cfg.get("reset_every_k", 1) or 1)
                if (self.timestep - self._rlast) >= _dwell and ((self._rcand - 1) % _everyk == 0):
                    import os as _osp_o
                    _pm_o = _osp_o.environ.get("MOE_DYN_PARTIAL", "").strip()
                    _done_o = False
                    if _pm_o:
                        try:
                            _done_o = self._dyn_partial_reset(_pm_o)
                        except Exception as _e:
                            print("[ORACLE_PARTIAL] failed (%s) -> fall back to full reset" % _e, flush=True)
                            _done_o = False
                    if not _done_o:
                        load_snapshot_cpu(self.slow_system, "init")
                    snapshot_cpu(self.slow_system, "start")
                    self.system.history["start"] = self.slow_system.history["start"]
                    self.memory.clear()
                    self._rlast = self.timestep
                    print("[ORACLE_RESET] at timestep=%d (partial=%s)" % (self.timestep, _pm_o if _done_o else "none"), flush=True)
            self._init_start(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            errs.append(wer(sample["text"], trans))
            transcriptions.append((sample["text"], trans))

            if oracle:
                self.system.model.set_forced_expert(None)

            self._update(sample)
            # [DYN_RESET] 标定（内部只在 K 窗口内做两遍 loss；未开启则直接返回）
            if self._dyn_on:
                try:
                    self._dyn_build(sample)
                except Exception as _e:
                    print("[DYN_RESET] build failed: %s" % _e, flush=True)
            self.timestep += 1

        print("#Too long: ", long_cnt)
        if getattr(self, "_leak_beta", 0.0) > 0:
            try:
                _isd = self.slow_system.history["init"][0]
                _num = 0.0
                _den = 0.0
                _cnt = 0
                for _ln, _lp in self.slow_system.model.named_parameters():
                    _lr = _isd.get(_ln)
                    if _lp.requires_grad and _lr is not None:
                        _lr = _lr.to(_lp.device)
                        _num += float((_lp.data - _lr).abs().mean())
                        _den += float(_lr.abs().mean())
                        _cnt += 1
                print("[LEAK_BETA] final drift mean|d|/mean|init|=%.4e over %d tensors beta=%.4f"
                      % (_num / max(_den, 1e-12), _cnt, self._leak_beta), flush=True)
            except Exception as _e:
                print("[LEAK_BETA] report failed: %s" % _e, flush=True)
        # ==================== [EXPERT_DELTA_REPORT] ====================
        try:
            for _tag, _sys in (("slow", self.slow_system), ("fast", self.system)):
                _rs = []
                for _m in getattr(_sys.model, "moe_layer", []):
                    try:
                        _ew = _m.experts_weight.data.float()
                        _bn = float(_m.weight.data.float().norm())
                        if _bn <= 0:
                            continue
                        for _e in range(_ew.shape[0]):
                            _rs.append(float(_ew[_e].norm()) / _bn)
                    except Exception:
                        pass
                if _rs:
                    _rs.sort()
                    _n = len(_rs)
                    print("[DELTA] %s | expert layersxE=%d | dW/W: mean=%.3e median=%.3e max=%.3e"
                          % (_tag, _n, sum(_rs) / _n, _rs[_n // 2], _rs[-1]), flush=True)
        except Exception as _e:
            print("[DELTA] stat failed: %s" % _e, flush=True)
        # [FS_SLOW_PATCH] 结束自检: 慢模型共享权重必须纹丝不动, 快模型必须动过
        try:
            _ref = getattr(self, "_fs_ref", None)
            if _ref and _ref.get("slow") is not None:
                _ds = float((self._shared_vec(self.slow_system) - _ref["slow"]).abs().max())
                _df = float((self._shared_vec(self.system) - _ref["fast"]).abs().max())
                print("[FS_SLOW_CHECK] slow_shared max|d|=%.3e (应=0) | fast_shared max|d|=%.3e (应>0)"
                      % (_ds, _df), flush=True)
        except Exception as _e:
            print("[FS_SLOW_CHECK] failed: %s" % _e, flush=True)
        # [FAST_MASK_PATCH] 结束自检
        try:
            if getattr(self, "_orig_step", None) is not None:
                print("[FAST_MASK_CHECK] excluded max|d|=%.3e (应=0) | selected max|d|=%.3e (应>0)"
                      % (self._fm_excl, self._fm_sel), flush=True)
        except Exception as _e:
            print("[FAST_MASK_CHECK] failed: %s" % _e, flush=True)
        if self._dyn_on:
            print("[DYN_RESET] total resets=%d" % self._dyn_n, flush=True)
        return {"wers": errs, "n_words": n_words,
                "transcriptions": transcriptions, "losses": losses}

    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.adapt_count


