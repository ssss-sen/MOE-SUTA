import math
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Callable

from .moe_conv import MoEConv1dLayer
from .moe_normalization import MoENormalizationLayer
from .moe_proj import MoELinearLayer
from .router import MLPRouter
from .utils import set_nested_attr
from .loss import softmax_entropy


class MoETTA(nn.Module):
    """
    以block为单位来决定分配的专家
    """

    def __init__(
        self,
        model,
        num_expert: int,
        topk: int,
        hidden_dim: int,
        dynamic_threshold: bool = True,
        dynamic_lb: bool = True,
        lb_coeff: float = 1.0,
        weight_by_prob: bool = True,
        weight_by_entropy: bool = True,
        randomness: float = 0,
        e_margin_coeff: float = 0.4,
        activate_shared_expert: bool = False,
        route_penalty: float = 0.0,
        decay: float = 0.0,
        self_router: bool = True,
        grad_hook: bool = False,
        disabled_layer: List[int] = None,
        normal_layer: List[int] = None,
        pass_through_coeff: bool = True,
        global_router_idx: int = -1,
        device: str = "cuda",
        frames_per_sec: int = 50,
        audio_sample_rate: int = 16000,
        samplewise: bool = True,
        lb_window: int = 1,          # [V1-lbwindow] batch 级 LB 的窗口
        batch_route: bool = False,   # [BR] batch 级路由（用跨样本统计量做专家决策）
        batch_route_momentum: float = 0.5,   # [BR] EMA 动量
        route_granularity: str = "native",   # [RG] native / sample / block
        expert_schedule: str = "",           # [ES] "1-8:9,11-59:4,64-107:1"
    ):
        """
        初始化

        Args:
            model: 用于TTA的模型
            num_expert: 专家数
            topk: 路由的专家数
            weight_by_prob: 是否将路由概率作为融合时的系数. Defaults to True.
            randomness: 随机初始化专家时模长和预训练参数模长之比. Defaults to 0.
            shared_expert: 是否有共享专家（这里把预训练参数作为共享专家）. Defaults to True.
            route_penalty: auxiliary loss free所用超参数. Defaults to 0.0.
            device: 所用设备. Defaults to "cuda".
        """
        super().__init__()
        assert num_expert >= topk, "num_expert must greater or equal than topk"
        assert route_penalty >= 0, "route penalty should be positive"
        self.device = device
        self.model = model.to(self.device)
        self.num_expert = num_expert
        if self_router and global_router_idx < 0:
            self.router = None
        else:
            self.router = MLPRouter(num_expert, hidden_dim, device)
            self.cnt = torch.zeros(self.num_expert).to(device)
        self.topk = topk
        self.weight_by_prob = weight_by_prob
        self.randomness = randomness
        self.activate_shared_expert = activate_shared_expert
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_lb = dynamic_lb
        self.weight_by_entropy = weight_by_entropy
        self.lb_coeff = lb_coeff
        self.route_penalty = route_penalty
        self.decay = decay
        self.step = 0
        self.self_router = self_router
        self.penalty = torch.zeros(self.num_expert).to(self.device)
        self.grad_hook = grad_hook
        self.disabled_layer = disabled_layer
        self.normal_layer = normal_layer
        self.pass_through_coeff = pass_through_coeff
        self.e_margin_coeff = e_margin_coeff
        self.criterion_mse = nn.MSELoss(reduction="none").cuda()
        self.global_router_idx = global_router_idx
        self.frames_per_sec = frames_per_sec
        self.audio_sample_rate = audio_sample_rate
        self.samplewise = samplewise
        self.lb_window_cfg = int(lb_window or 1)   # [V1-lbwindow]
        import os as _os_lbg
        self.lb_global = int(_os_lbg.environ.get("MOE_LB_GLOBAL", "0") or 0) > 0
        if self.lb_global:
            print("[LB_GLOBAL] load 改用 self.cnt（自流开始累积）| importance 仍为本次 forward", flush=True)
        self.batch_route_cfg = bool(batch_route)             # [BR]
        self.br_momentum_cfg = float(batch_route_momentum)   # [BR]
        self.rg_cfg = str(route_granularity or "native")     # [RG]
        # [ES] 分层专家数：形如 "1-8:9,11-59:4,64-107:1"（idx 段:专家数），未命中段用 num_expert
        self.es_cfg = str(expert_schedule or "")
        self._es_segments = []
        for seg in self.es_cfg.split(","):
            seg = seg.strip()
            if not seg or ":" not in seg:
                continue
            rng, _, val = seg.partition(":")
            try:
                e = int(val)
            except ValueError:
                continue
            parts = rng.split("-")
            try:
                a = int(parts[0])
                b = int(parts[1]) if len(parts) > 1 else a
            except ValueError:
                continue
            self._es_segments.append((a, b, e))
        self.normal_layers = []
        self.output_list = []
        self.entropys = []
        self.probs = []
        self.biased_probs = []
        # 日志/统计防泄漏：默认不记录；可视化调试时再置 keep_log=True
        self.keep_log = False
        self._ent_sum = 0.0
        self._ent_n = 0
        self.cnt = torch.zeros(self.num_expert).to(self.device)
        self.filtered_cnt = 0
        self.construct_model()
        self.criterion_mse = nn.MSELoss(reduction="none").to(self.device)

    def construct_model(self):
        """
        Replace Normalization with MoENormalization
        """

        def router_hook(module, input):
            coeff, topks, biased_prob, route_prob = self.get_coeff_topks(input[0])
            self.biased_prob = biased_prob
            self.route_prob = route_prob
            self.set_coeff(coeff)
            self.set_topks(topks)

        self.model.requires_grad_(False)
        idx = 0
        current_fps = float(self.audio_sample_rate)   # raw 波形 16000 samples/s

        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Conv1d):
                if idx in self.disabled_layer and idx in self.normal_layer:
                    raise ValueError(f"{idx} is both in disabled layer and normal layer")
                stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
                if idx in self.disabled_layer:
                    idx += 1
                    current_fps = current_fps / stride   # disabled 层也降低分辨率
                    continue
                if idx in self.normal_layer:
                    idx += 1
                    mod.requires_grad_(True)
                    self.normal_layers.append(mod)
                    current_fps = current_fps / stride
                    continue

                new_mod = MoEConv1dLayer(
                    idx=idx,
                    num_expert=self._expert_for_idx(idx),   # [ES] 分层专家数
                    activate_shared_expert=self.activate_shared_expert,
                    base_mod=mod,
                    randomness=self.randomness,
                    self_router=True,  # 卷积层自带 Router
                    samplewise=self.samplewise,
                    topk=self.topk,
                    weight_by_prob=self.weight_by_prob,
                    penalty=self.route_penalty,
                    decay=self.decay,
                    device=self.device,
                    grad_hook=self.grad_hook,
                    pass_through_coeff=self.pass_through_coeff,
                    frames_per_sec=int(current_fps),   # 该层输入每秒样本数 → 每块≈1秒
                )
                set_nested_attr(self.model, name, new_mod)
                idx += 1
                current_fps = current_fps / stride

            elif isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if idx in self.disabled_layer and idx in self.normal_layer:
                    raise ValueError(f"{idx} is both in disabled layer and normal layer")
                if idx in self.disabled_layer or idx < self.global_router_idx:
                    idx += 1
                    continue
                if idx in self.normal_layer:
                    idx += 1
                    mod.requires_grad_(True)
                    self.normal_layers.append(mod)
                    continue
                new_mod = MoENormalizationLayer(
                    idx=idx,
                    num_expert=self._expert_for_idx(idx),   # [ES] 分层专家数
                    activate_shared_expert=self.activate_shared_expert,
                    base_mod=mod,
                    randomness=self.randomness,
                    self_router=self.self_router,
                    samplewise=self.samplewise,
                    topk=self.topk,
                    weight_by_prob=self.weight_by_prob,
                    penalty=self.route_penalty,
                    decay=self.decay,
                    device=self.device,
                    grad_hook=self.grad_hook,
                    pass_through_coeff=self.pass_through_coeff,
                    frames_per_sec=int(current_fps),   # [RG] 该层输入每秒 token 数（逐块路由用）
                )
                if idx == self.global_router_idx:
                    new_mod.register_forward_pre_hook(router_hook)
                set_nested_attr(self.model, name, new_mod)
                idx += 1

            elif isinstance(mod, nn.Linear):
                # BUG-004 决策 B-2：Linear 保持为原始共享层，不 MoE 化。
                # 理由：MoETTA 论文只专家化 LayerNorm；本仓库 SUTA 基线也只更新 feature conv+LN、
                # encoder 线性权重冻结 —— 因此 MoE-SUTA 与 SUTA 的可比差异应落在 LN/conv 的专家增量上。
                # 若需要放开某些 Linear 作为普通可训练层，把其 idx 配进 normal_layer 即可。
                if idx in self.normal_layer:
                    idx += 1
                    mod.requires_grad_(True)
                    self.normal_layers.append(mod)
                    continue
                idx += 1
                continue
        self.moe_layer = [
            mod
            for mod in self.model.modules()
            if isinstance(mod, (MoENormalizationLayer, MoEConv1dLayer, MoELinearLayer))
        ]

        # [ES]/[RG]/[BR] 自检：让日志一眼看出这些实验开关有没有生效
        if self._es_segments or self.normal_layer:
            _ec = {}
            for _m in self.moe_layer:
                _ec[_m.num_expert] = _ec.get(_m.num_expert, 0) + 1
            print("[ES] schedule=%s -> %s | normal_layer=%s (%d 个普通可训练层) | MoE 层 E 分布=%s"
                  % (self.es_cfg, self._es_segments, self.normal_layer,
                     len(self.normal_layers), _ec), flush=True)
        if self.rg_cfg != "native":
            print("[RG] route_granularity=%s" % self.rg_cfg, flush=True)

        # 实验用：MOE_FORCE_RANDOM=1 时强制随机路由（归因消融）
        if os.environ.get("MOE_FORCE_RANDOM", "0") == "1":
            for m in self.moe_layer:
                m.force_random = True

        # [V1-lbwindow] 注入 batch 级 LB 窗口（默认 1 = 旧行为）
        for m in self.moe_layer:
            if hasattr(m, "lb_window"):
                m.lb_window = self.lb_window_cfg
        
        # [BR] 注入 batch 级路由开关（默认 False = 逐样本路由，与历史结果一致）
        for m in self.moe_layer:
            m.batch_route = self.batch_route_cfg
            if hasattr(m, "br_momentum"):
                m.br_momentum = self.br_momentum_cfg
            if hasattr(m, "route_granularity"):
                m.route_granularity = self.rg_cfg

        # ===================== [FUSE_MODE_PATCH] =====================
        # 目的：让"融合系数"可以有第二种语义（官方 MoETTA 是 E 维全非零的 c=prob/prob.detach()），
        #      而 topk 仍固定为 1（师兄要求：输出固定、方便算指标）。
        # 门控：环境变量 MOE_FUSE_MODE（topk 默认 = 完全不改变原有行为）
        #   sum  ⇒ c_e ≡ 1，融合 = Σ_e ΔW_e（所有专家 delta 之和；前向与"选了谁"无关）
        #   prob ⇒ c_e = prob_e，融合 = Σ_e prob_e·ΔW_e（概率加权混合，置信度参与）
        self.fuse_mode_cfg = os.environ.get("MOE_FUSE_MODE", "topk").strip().lower()
        # [FUSE_SCALE_PATCH] 融合 delta 的整体幅度缩放（默认 1.0 ⇒ 与历史结果完全一致）
        #   动机：prob(凸组合，幅度≈单个 delta) 在 md_easy 最好、sum(等权全加，幅度≈3~9×) 在 md_hard 最好
        #   ⇒ 说明"融合 delta 的有效幅度"是有效自由度 ⇒ 扫 s ∈ {0.3, 1, 3} 找两个数据集各自的最优点
        self.fuse_scale_cfg = float(os.environ.get("MOE_FUSE_SCALE", "1.0"))
        # [FUSE_TOPK_PATCH] 融合侧 TopK（0 = 不启用）；env 门控，默认 0 ⇒ 行为与历史完全一致
        self.fuse_topk_cfg = int(os.environ.get("MOE_FUSE_TOPK", "0") or 0)
        for m in self.moe_layer:
            m.fuse_mode = self.fuse_mode_cfg
            m.fuse_scale = self.fuse_scale_cfg
            m.fuse_topk = self.fuse_topk_cfg
        if self.fuse_mode_cfg != "topk" or self.fuse_scale_cfg != 1.0 or self.fuse_topk_cfg > 0:
            if self.rg_cfg == "block":
                print("[FUSE] 警告：route_granularity=block 未实现 fuse_mode（该层仍走 topk 语义）", flush=True)
            if self.fuse_topk_cfg > 0:
                _who = ("融合侧 TopK=%d（TopK 内按 prob 重新归一化；_fk=1 等价 topk、_fk=E 等价 prob）"
                        % self.fuse_topk_cfg)
            elif self.fuse_mode_cfg in ("sum", "prob", "mean"):
                _who = "融合改用全部 %d 个专家（c_e 语义=%s）" % (self.num_expert, self.fuse_mode_cfg)
            else:
                _who = "融合仍只用 topk=%d 选中的 1 个专家（仅 delta 幅度 x%.3f）" % (
                    self.topk, self.fuse_scale_cfg)
            print("[FUSE] mode=%s | scale=%.3f | fuse_topk=%d | topk 仍为 %d | %s"
                  % (self.fuse_mode_cfg, self.fuse_scale_cfg, self.fuse_topk_cfg, self.topk, _who), flush=True)

        # ==================== [EXPERT_INIT_PATCH] ====================
        # 目的（师兄方案 a）：让"专家 = 原参数的一份完整拷贝"（而不是"零初始化的增量"），
        #   并配合 MOE_NO_SHARED=1（融合时【不叠加基座】）⇒ t=0 输出 = 原模型，之后每个专家各自分化。
        #   ⚠️ 只做初始化而不设 no_shared ⇒ t=0 变成 2×W_base（LayerNorm scale 1→2、卷积权重翻倍）⇒ 必崩。
        # 门控：MOE_EXPERT_INIT=base、MOE_NO_SHARED=1（默认都不设 ⇒ 行为与历史结果完全一致）
        self.no_shared_cfg = os.environ.get("MOE_NO_SHARED", "0") == "1"
        for m in self.moe_layer:
            m.no_shared = self.no_shared_cfg
        self.expert_init_cfg = os.environ.get("MOE_EXPERT_INIT", "").strip().lower()
        if self.expert_init_cfg == "base":
            _maxdiff, _nb = 0.0, 0
            for m in self.moe_layer:
                m.experts_weight.data.copy_(
                    m.weight.data.unsqueeze(0).expand_as(m.experts_weight))
                _maxdiff = max(_maxdiff, float(
                    (m.experts_weight.data[0] - m.weight.data).abs().max().item()))
                try:
                    if getattr(m, "bias", None) is not None and m.bias.numel() > 0:
                        m.experts_bias.data.copy_(
                            m.bias.data.unsqueeze(0).expand_as(m.experts_bias))
                        _nb += 1
                except Exception as _e:
                    print("[EXPERT_INIT] bias 跳过（%s）: %s" % (type(m).__name__, _e), flush=True)
            print("[EXPERT_INIT] base | 层数=%d | 专家[0] 与基座最大逐元素差=%.3e | bias 同步层=%d | no_shared=%s"
                  % (len(self.moe_layer), _maxdiff, _nb, self.no_shared_cfg), flush=True)
        if self.no_shared_cfg:
            print("[EXPERT_INIT] MOE_NO_SHARED=1 ⇒ 融合【替换】基座；多专家请用 prob/mean（sum 会放大 E 倍）", flush=True)

        # ==================== [RANDOMNESS_SELFCHECK] ====================
        # randomness 此前没有任何自检 ⇒ 无法从日志确认开关生效。这里直接量"专家[0]/基座"的模长比。
        _rr = []
        for _m in list(getattr(self, "moe_layer", []))[:3]:
            try:
                _ew = _m.experts_weight.data[0].float()
                _bw = _m.weight.data.float()
                _rr.append(float(_ew.norm() / _bw.norm().clamp(min=1e-12)))
            except Exception:
                pass
        if _rr:
            print("[RANDOM] randomness=%.4f | 实测 专家[0]/基座 模长比 = %s | 期望 = %.4f"
                  % (self.randomness, " ".join("%.3e" % _r for _r in _rr), self.randomness), flush=True)

        # ==================== [SHARED_ROUTER_PATCH] ====================
        # 目的：任务③"共享路由"——让同类型同维度的层共用一个 router（默认每层独立）。
        # 门控：环境变量 MOE_SHARED_ROUTER=1（默认 0 = 每层独立 router，行为不变）
        # 实现：直接把这些层的 .router 指向同一个 nn.Linear（forward 路径完全不用改）；
        #       分组键 = (基础模块类型, 输入维度) ⇒ encoder 的 768 维 LN 共一组、前端 conv 共一组。
        self.shared_router_cfg = os.environ.get("MOE_SHARED_ROUTER", "0") == "1"
        if self.shared_router_cfg:
            _pool, _cnt = {}, {}
            for m in self.moe_layer:
                r = getattr(m, "router", None)
                if r is None:
                    continue
                # [SHARED_ROUTER_PATCH_V2] 分组键必须包含 num_expert！
                #   expert_schedule([ES]) 会让各层 E 不同（例如 1-8:4,11-48:4 ⇒ E=4），
                #   而模型级 self.num_expert 可能仍是 9 ⇒ 用全局 E 建共享 router 会导致
                #   prob(9) 与 self.penalty(4) 广播失败（V1 的崩溃原因）。
                #   防御：router 输出维度与该层 num_expert 不一致时跳过该层并警告。
                if int(r.out_features) != int(m.num_expert):
                    print("[SHARED_ROUTER] 跳过 idx=%s（router out=%d != num_expert=%d）"
                          % (getattr(m, "idx", "?"), r.out_features, m.num_expert), flush=True)
                    continue
                key = (type(getattr(m, "base_mod", m)).__name__, int(r.in_features), int(m.num_expert))
                if key not in _pool:
                    _lr = nn.Linear(key[1], key[2], device=self.device)
                    nn.init.xavier_normal_(_lr.weight)
                    nn.init.zeros_(_lr.bias)
                    _pool[key] = _lr
                m.router = _pool[key]
                _cnt[key] = _cnt.get(key, 0) + 1
            self._shared_router_pool = _pool
            print("[SHARED_ROUTER] 共享组=%s | 被共享的层=%d/%d | 独立 router 数=%d"
                  % ({"%s:%d:E%d" % k: _cnt[k] for k in _pool},
                     sum(_cnt.values()), len(self.moe_layer), len(_pool)), flush=True)

    def _expert_for_idx(self, idx: int) -> int:
        """[ES] 返回该层应使用的专家数（未命中 expert_schedule 时用全局 num_expert）"""
        for a, b, e in getattr(self, "_es_segments", []):
            if a <= idx <= b:
                return max(1, int(e))
        return self.num_expert

    def get_coeff_topks(self, embedding):
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

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if not (self.training or getattr(self, "collect_stats", False)):
            return output
        logits = output.logits if hasattr(output, 'logits') else output
        lb_loss = self.collect_lb_loss().sum()
        temp = getattr(self, "current_temp", 1.0)
        entropy = softmax_entropy(logits / temp)
        self.current_entropy = entropy
        if self.weight_by_entropy:
            # detach() 很重要，防止梯度流过权重系数
            _e0 = float(os.environ.get("MOE_E0", "0") or 0.0)
            _nc = float(os.environ.get("MOE_E0_CLASSES", "0") or 0.0)
            if _e0 <= 0:
                # [MOETTA_FILTER] E_0 可配：默认保持官方写死的 e_margin_coeff*ln(1000)
                _e0 = (self.e_margin_coeff * math.log(_nc)) if _nc > 0 else (self.e_margin_coeff * math.log(1000))
            self.current_coeff = 1 / (torch.exp(entropy.detach() - _e0))
        else:
            self.current_coeff = 1.0  # 标量
        if self.keep_log:
            self.output_list.append(logits.detach().cpu())

        # 防泄漏：累计均值替代只增不清的 entropys 列表（new/old = 全体样本平均熵 更新后/更新前）
        mean_entropy = entropy.detach().mean()
        mean_entropy_f = float(mean_entropy.item())
        if self._ent_n == 0:
            self._ent_n = 1
            self._ent_sum = mean_entropy_f
            if self.dynamic_threshold:
                self.threshold = mean_entropy
            if self.dynamic_lb:
                self.lb_coeff = mean_entropy * self.lb_coeff
        else:
            old_avg = self._ent_sum / self._ent_n
            self._ent_sum += mean_entropy_f
            self._ent_n += 1
            new_avg = self._ent_sum / self._ent_n
            ratio = new_avg / old_avg
            if self.dynamic_threshold:
                self.threshold = self.threshold * ratio
            if self.dynamic_lb:
                self.lb_coeff = self.lb_coeff * ratio
        if not hasattr(self, "threshold"):
            self.threshold = mean_entropy
        valid_mask = (entropy <= self.threshold)
        entropy_filtered = entropy[valid_mask]
        self.valid_mask = valid_mask

        if entropy_filtered.numel() > 0:
            self.moe_lb_loss = self.lb_coeff * lb_loss
        else:
            # 防止全部被过滤导致计算图断裂 (加上 0 * logits)
            self.moe_lb_loss = self.lb_coeff * lb_loss + 0.0 * logits.sum()

        self.filtered_cnt += logits.numel() // logits.shape[-1] - entropy_filtered.numel()
        self.step_once()

        if not self.self_router:
            if self.keep_log:
                self.probs.append(self.route_prob)
                self.biased_probs.append(self.biased_prob)
            route_prob_max, selected = torch.topk(self.biased_prob, self.topk)
            self.cnt[selected] += 1
            
        return output

    def set_coeff(self, coeff: torch.Tensor):
        """
        set current expert
        """
        for layer in self.moe_layer:
            layer.update_coeff(coeff)

    def set_topks(self, topks: torch.Tensor):
        """
        set topks
        """
        for layer in self.moe_layer:
            layer.topks = topks

    def get_expert(self, experts) -> List[nn.Parameter]:
        """
        get current expert
        """
        params = []
        for mod in self.moe_layer:
            for exp in experts:
                if hasattr(mod, 'experts_weight'):
                    params.append(mod.experts_weight[exp])
                # 注意：Conv1d 可能没有 bias，空 tensor.numel() 会是 0
                if hasattr(mod, 'experts_bias') and mod.experts_bias.numel() > 0:
                    params.append(mod.experts_bias[exp])
        return params

    def get_shared_expert(self) -> List[nn.Parameter]:
        """
        get shared expert, i.e., the pretrained parameter of LN
        """
        params = []
        for mod in self.moe_layer:
            if hasattr(mod, 'weight'):
                params.append(mod.weight)
            if hasattr(mod, 'bias') and mod.bias is not None and mod.bias.numel() > 0:
                params.append(mod.bias)
        return params

    def collect_lb_loss(self):
        if not self.self_router:
            return self.lb_loss
        loss = []
        for mod in self.moe_layer:
            # [FIX-16] train 模式下 wav2vec2 的 layerdrop=0.1 会整层跳过 encoder 层，
            # 被跳过层里的 MoE 归一化子模块没有前向 → lb_loss 仍是 None，
            # 旧代码 torch.stack 会因 NoneType 直接崩；这里只收集已产生统计量的层。
            if getattr(mod, "lb_loss", None) is not None:
                loss.append(mod.lb_loss)
        if len(loss) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(loss)

    def set_batch_route(self, flag: bool):
        """[BR] 运行时开关 batch 级路由"""
        self.batch_route_cfg = bool(flag)
        for m in self.moe_layer:
            m.batch_route = self.batch_route_cfg

    def set_forced_expert(self, k):
        """[ORACLE] 强制所有 MoE 层使用第 k 个专家（k=None 恢复正常路由）。

        用于"专家=域"的上限实验：把每个噪声域固定绑到一个专属专家上，
        测出"如果路由是完美的 oracle，MoE 最好能到多少"。
        """
        for m in self.moe_layer:
            m.force_expert = None if k is None else int(k)

    def update_batch_stats(self):
        """[BR] 策略在每个样本适应结束时调用：把该样本的统计量并入各层的 EMA"""
        for m in self.moe_layer:
            if hasattr(m, "update_batch_stats"):
                m.update_batch_stats()

    def reset_batch_stats(self):
        """[BR] 策略在每条流开始时调用：清空各层的 EMA"""
        for m in self.moe_layer:
            if hasattr(m, "reset_batch_stats"):
                m.reset_batch_stats()

    def step_once(self):
        for mod in self.moe_layer:
            mod.step_once()
        self.step += 1

    def get_params(self):
        params = []
        if self.router is not None:
            params.extend(self.router.get_params())
        for mod in self.model.modules():
            if isinstance(mod, (MoENormalizationLayer, MoEConv1dLayer)):
                params.extend(mod.get_trainable_params())
        for m in self.normal_layers:
            for name, p in m.named_parameters():
                if name in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
        # ==================== [FREEZE_SHARED_PATCH] ====================
        # 背景（2026-09-20 发现）：build_optimized_model() 会对 get_params() 返回的每个 p 设
        #   requires_grad=True，**覆盖掉 activate_shared_expert=False** ⇒ 共享(基座)专家其实一直在被训练，
        #   配置里那个"冻结共享专家"的开关是失效的。这也是"delta 版单专家 ≈ dsuta"的原因。
        # 门控 MOE_FREEZE_SHARED=1（默认不设 ⇒ 行为与历史结果完全一致）：
        #   把 MoE 层的 weight/bias（共享基座）从可训练列表里剔除 ⇒ 真正冻结；normal_layer 的权重不受影响。
        self.freeze_shared_cfg = os.environ.get("MOE_FREEZE_SHARED", "0") == "1"
        if self.freeze_shared_cfg:
            _excl = set()
            for mod in self.model.modules():
                if isinstance(mod, (MoENormalizationLayer, MoEConv1dLayer, MoELinearLayer)):
                    for _n in ("weight", "bias"):
                        _t = getattr(mod, _n, None)
                        if _t is not None:
                            _excl.add(id(_t))
            _before = len(params)
            params = [p for p in params if id(p) not in _excl]
            _rg = 0
            for mod in self.model.modules():
                if isinstance(mod, (MoENormalizationLayer, MoEConv1dLayer, MoELinearLayer)):
                    for _n in ("weight", "bias"):
                        _t = getattr(mod, _n, None)
                        if _t is not None and _t.requires_grad:
                            _rg += 1
            print("[FREEZE_SHARED] 冻结共享(基座)专家：参数 %d → %d（剔除 %d 个）；"
                  "残留 requires_grad=True 的共享张量=%d（应为 0）"
                  % (_before, len(params), _before - len(params), _rg), flush=True)
        # [SHARED_ROUTER_PATCH] 共享 router 会让同一个 Parameter 在各层重复出现 ⇒ 必须去重，
        # 否则 optimizer 会对它每步更新多次（等效放大 lr）。**只在共享模式下去重**，
        # 保证默认路径（不共享）与历史结果字节级一致。
        if getattr(self, "shared_router_cfg", False):
            _seen, _out = set(), []
            for p in params:
                if id(p) in _seen:
                    continue
                _seen.add(id(p))
                _out.append(p)
            print("[SHARED_ROUTER] 参数去重：%d -> %d" % (len(params), len(_out)), flush=True)
            return _out
        return params