import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple


@torch.jit.script
def fuse_params(
    experts_weight: torch.Tensor,
    experts_bias: torch.Tensor,
    topks: torch.Tensor,
    coeff: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_bias: torch.Tensor,
    add_shared: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, K = topks.shape
    D = experts_weight.size(1)

    experts_weight = experts_weight.unsqueeze(0).unsqueeze(0)
    experts_bias = experts_bias.unsqueeze(0).unsqueeze(0)

    experts_weight = experts_weight.expand(B, T, -1, -1)
    experts_bias = experts_bias.expand(B, T, -1, -1)
    expanded_topks = topks.unsqueeze(-1).expand(-1, -1, -1, D)

    selected_weight = experts_weight.gather(2, expanded_topks)
    selected_bias = experts_bias.gather(2, expanded_topks)

    selected_coeff = coeff.gather(2, topks).unsqueeze(-1)

    fused_weight = (selected_weight * selected_coeff).sum(2)
    fused_bias = (selected_bias * selected_coeff).sum(2)

    # [EXPERT_INIT_PATCH] add_shared=False ⇒ 融合结果【替换】基座（= 专家是完整权重）
    if add_shared:
        final_weight = fused_weight + shared_weight.view(1,1,-1)
        final_bias = fused_bias + shared_bias.view(1,1,-1)
    else:
        final_weight = fused_weight
        final_bias = fused_bias

    return final_weight, final_bias


@torch.jit.script
def apply_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if x.dim() == 3:
        if weight.dim() == 2:
            return x_norm * weight.unsqueeze(1) + bias.unsqueeze(1)
        elif weight.dim() == 3:
            return x_norm * weight + bias
        else:
            raise NotImplementedError
    elif x.dim() == 4:
        # [B, H, W, D]
        return x_norm * weight.unsqueeze(1).unsqueeze(1) + bias.unsqueeze(1).unsqueeze(
            1
        )
    else:
        raise NotImplementedError

@torch.jit.script
def fuse_params_samplewise(
    experts_weight: torch.Tensor,   # (E, D)
    experts_bias: torch.Tensor,     # (E, D)
    topks: torch.Tensor,            # (B, K)
    coeff: torch.Tensor,            # (B, E)
    shared_weight: torch.Tensor,    # (D,)
    shared_bias: torch.Tensor,      # (D,)
    add_shared: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = topks.shape
    D = experts_weight.size(1)
    expanded_w = experts_weight.unsqueeze(0).expand(B, -1, -1)
    sel_w = expanded_w.gather(1, topks.unsqueeze(-1).expand(B, K, D))
    c = coeff.gather(1, topks).unsqueeze(-1)                 # (B,K,1)
    # [EXPERT_INIT_PATCH] add_shared=False ⇒ 融合结果【替换】基座
    if add_shared:
        final_w = (sel_w * c).sum(1) + shared_weight.unsqueeze(0)  # (B,D)
        expanded_b = experts_bias.unsqueeze(0).expand(B, -1, -1)
        sel_b = expanded_b.gather(1, topks.unsqueeze(-1).expand(B, K, D))
        final_b = (sel_b * c).sum(1) + shared_bias.unsqueeze(0)    # (B,D)
    else:
        final_w = (sel_w * c).sum(1)
        expanded_b = experts_bias.unsqueeze(0).expand(B, -1, -1)
        sel_b = expanded_b.gather(1, topks.unsqueeze(-1).expand(B, K, D))
        final_b = (sel_b * c).sum(1)
    return final_w, final_b
@torch.jit.script
def apply_batchnorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    B, C, H, W = x.shape
    # 按 (B, C, H, W) → (B, C, H*W)，然后在(B, HW)上做均值/方差
    x_flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)

    mean = x_flat.mean(dim=1)
    var = x_flat.var(dim=1, unbiased=False)

    mean = mean.view(1, C, 1, 1)
    var = var.view(1, C, 1, 1)

    x_norm = (x - mean) / torch.sqrt(var + eps)

    weight = weight.view(B, C, 1, 1)
    bias = bias.view(B, C, 1, 1)

    out = x_norm * weight + bias
    return out


@torch.jit.script
def apply_groupnorm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if x.dim() == 3:
        # 3D 输入 (B, C, L)：wav2vec2 GroupNormConvLayer 的形态
        B, C, L = x.shape
        G = num_groups
        assert C % G == 0, "channels must be divisible by num_groups"
        xr = x.view(B, G, -1)
        mean = xr.mean(dim=-1, keepdim=True)
        var = xr.var(dim=-1, unbiased=False, keepdim=True)
        xn = (xr - mean) / torch.sqrt(var + eps)
        xn = xn.view(B, C, L)
        if weight.dim() == 2:          # 逐样本融合权重 (B,C)
            return xn * weight.unsqueeze(-1) + bias.unsqueeze(-1)
        elif weight.dim() == 1:        # 共享/退化权重 (C,)
            return xn * weight.view(1, C, 1) + bias.view(1, C, 1)
        else:
            raise NotImplementedError("groupnorm 3D: unsupported weight rank")
    elif x.dim() == 4:
        # 原 4D (B,C,H,W) 逻辑保留
        B, C, H, W = x.shape
        G = num_groups
        assert C % G == 0, "channels must be divisible by num_groups"
        xv = x.view(B, G, -1)
        mean = xv.mean(dim=-1, keepdim=True)
        var = xv.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (xv - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(B, C, H, W)
        if weight.dim() == 2:
            weight = weight.view(B, C, 1, 1)
            bias = bias.view(B, C, 1, 1)
        elif weight.dim() == 1:
            weight = weight.view(1, C, 1, 1)
            bias = bias.view(1, C, 1, 1)
        else:
            raise NotImplementedError("groupnorm 4D: unsupported weight rank")
        return x_norm * weight + bias
    else:
        raise NotImplementedError(f"unsupported x.dim()={x.dim()}")


class MoENormalizationLayer(nn.Module):
    """
    通用 MoE 归一化层基类，封装专家参数、路由器以及参数融合逻辑。
    可根据 base_mod 类型自动调用对应的归一化实现。
    """

    def __init__(
        self,
        idx: int,
        num_expert: int,
        activate_shared_expert: bool,
        base_mod: nn.Module,
        randomness: float,
        self_router: bool = False,
        samplewise: bool = None,
        topk: int = None,
        weight_by_prob: bool = None,
        penalty: float = None,
        decay: float = None,
        grad_hook: bool = False,
        pass_through_coeff: bool = True,
        device: str = "cuda",
        frames_per_sec: int = 50,     # [RG] 该层输入每秒 token 数（逐块路由用）
    ):
        super().__init__()
        assert num_expert > 0, "专家数必须大于 0"
        self.idx = idx
        self.num_expert = num_expert
        self.base_mod = base_mod
        self.self_router = self_router

        # 从传入的 base_mod 提取基础参数
        self.weight = base_mod.weight
        self.weight.requires_grad_(activate_shared_expert)
        self.bias = base_mod.bias
        self.bias.requires_grad_(activate_shared_expert)

        # 初始化专家参数
        weight_shape = self.weight.shape
        bias_shape = self.bias.shape
        self.experts_weight = nn.Parameter(
            torch.randn((num_expert,) + tuple(weight_shape), device=device)
        )
        self.experts_bias = nn.Parameter(
            torch.randn((num_expert,) + tuple(bias_shape), device=device)
        )
        # 缩放专家参数
        w_norm = torch.norm(self.weight).item()
        b_norm = torch.norm(self.bias).item()
        expert_w_norm = torch.norm(self.experts_weight.view(num_expert, -1), dim=1)
        expert_b_norm = torch.norm(self.experts_bias.view(num_expert, -1), dim=1)
        factor_w = (randomness * w_norm / expert_w_norm).view(
            num_expert, *([1] * len(weight_shape))
        )
        factor_b = (randomness * b_norm / expert_b_norm).view(
            num_expert, *([1] * len(bias_shape))
        )
        self.experts_weight.data *= factor_w
        self.experts_bias.data *= factor_b

        # 路由器
        if self_router:
            in_features = self.weight.numel()
            self.router = nn.Linear(in_features, num_expert, device=device)
            nn.init.xavier_normal_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        else:
            self.router = None
        # 路由配置
        self.samplewise = samplewise
        self.topk = topk
        self.weight_by_prob = weight_by_prob
        self.penalty = torch.zeros(num_expert, device=device)
        self.route_penalty = penalty
        self.decay = decay
        self.step = 0
        self.coeff = None
        self.topks = None
        self.pass_through_coeff = pass_through_coeff
        self.cnt = torch.zeros(num_expert, device=device)
        self.lb_loss = None
        # [V1-lbwindow] batch 级 LB（lb_window=1 时等于旧行为）
        self.lb_window = 1
        self._lb_imp = None
        self._lb_cnt = None
        self._lb_n = 0
        self.route_prob_list = []
        self.record_route = False
        self.force_random = False
        # [BR] batch 级路由：用"跨样本统计量"(EMA) 做专家决策，而不是只靠当前样本
        self.batch_route = False
        self.br_momentum = 0.5
        self._br_ema = None      # 已并入的历史统计量 (1, D)，detach
        self._br_acc = None      # 本样本累积量
        self._br_n = 0
        # [ORACLE] 强制专家号（None = 正常路由）；用于"专家=域"的上限实验
        self.force_expert = None
        # [RG] 路由粒度："native"(逐样本) / "block"(逐块，约 1 秒一块)
        self.route_granularity = "native"
        self.frames_per_sec = int(frames_per_sec)

        self.cls_feature = None

        # 可选梯度钩子注册略
        if grad_hook:
            hook_fn = self.make_step_aware_hook(f"layer{self.idx}", self)
            self.register_expert_block_hook(
                self.experts_weight, kind="weight", idx=self.idx, hook_fn=hook_fn
            )
            self.register_expert_block_hook(
                self.experts_bias, kind="bias", idx=self.idx, hook_fn=hook_fn
            )
            if self.router is not None:
                self.register_router_separate_hooks(
                    self.router, f"layer{self.idx}/router", hook_fn
                )

    @staticmethod
    def make_step_aware_hook(name_prefix, instance):
        def hook_fn(name, norm_tensor):
            full_name = f"{name_prefix}/{name}"

        return hook_fn

    @staticmethod
    def register_expert_block_hook(param, kind: str, idx: int, hook_fn: Callable):
        """
        给 expert block 的整体参数注册一个 hook，记录所有有梯度专家的最大值和均值。

        Args:
            param: nn.Parameter, shape [num_expert, ...]
            kind: "weight" or "bias"
            idx: 当前 LayerNorm 层的 idx
            hook_fn: Callable(name: str, value: Tensor) → Tensor
        """

        def hook(grad):
            grad_flat = grad.view(grad.size(0), -1)  # [num_expert, D]
            # 只保留有梯度的行（非零行）
            mask = grad_flat.abs().sum(dim=1) > 0  # [num_expert] boolean
            active_grads = grad_flat[mask]

            if active_grads.numel() > 0:
                norms = active_grads.norm(p=2, dim=1)  # [num_active_expert]
                max_norm = norms.max()
                mean_norm = norms.mean()
                hook_fn(f"layer{idx}/expert_{kind}_grad_max", max_norm)
                hook_fn(f"layer{idx}/expert_{kind}_grad_mean", mean_norm)

            return grad

        param.register_hook(hook)

    @staticmethod
    def register_router_separate_hooks(
        router: nn.Linear, name_prefix: str, hook_fn: Callable
    ):
        """
        为 router 的 weight 和 bias 分别注册梯度 hook，登记各自模长。

        Args:
            router: nn.Linear 实例
            name_prefix: 如 "layer0/router"
            hook_fn: Callable(name: str, norm_tensor: Tensor)
        """
        router.weight.register_hook(
            lambda grad: hook_fn(f"{name_prefix}_weight_grad_norm", grad.norm(2))
        )
        router.bias.register_hook(
            lambda grad: hook_fn(f"{name_prefix}_bias_grad_norm", grad.norm(2))
        )

    def get_trainable_params(self):
        params = [self.experts_bias, self.experts_weight]
        params.append(self.weight)
        params.append(self.bias)
        if hasattr(self,"router"):
            for name, p in self.router.named_parameters():
                params.append(p)
        return params

    def update_coeff(self, coeff: torch.Tensor):
        self.coeff = coeff if self.pass_through_coeff else coeff.detach()

    def get_topks(self, x: torch.Tensor):
        # [SAMPLE_LEVEL_ROUTE_C] 逐块路由已废弃：不论 route_granularity 取什么值，一律走下面的逐样本路径。
        # （原 `if rg == "block": self._get_topks_block(x); return` 已移除）
        B = x.size(0)
        if isinstance(self.base_mod, nn.LayerNorm):
            # 对于ViT（Patch Embedding后），每个token向量直接flatten
            dims = tuple(i for i in range(1, x.ndim - 1))
            flat = x.mean(dim=dims)
        elif isinstance(self.base_mod, nn.BatchNorm2d) or isinstance(
            self.base_mod, nn.GroupNorm
        ):
            # 对于ResNet（Feature Map），每个channel spatial average
            flat = x.mean(dim=tuple(range(2, x.dim())))  # (B, C), spatial avg
        else:
            raise ValueError(f"Unknown architecture {self.arch}")
        if self.samplewise:
            # [BR] batch 级路由（可选）：决策输入 = 历史样本统计量的 EMA；
            #      第 1 条样本（EMA 还是空）自动退化为逐样本路由。
            route_in = flat
            if getattr(self, "batch_route", False):
                cur = flat.detach().mean(0, keepdim=True)                      # (1, D)
                self._br_acc = cur if self._br_acc is None else self._br_acc + cur
                self._br_n += 1
                if self._br_ema is not None:
                    route_in = self._br_ema.expand(flat.shape[0], -1)
            prob = F.softmax(self.router(route_in), dim=-1)
            biased = prob - self.penalty
            fe = getattr(self, "force_expert", None)
            if fe is not None:
                # [ORACLE] 强制使用第 fe 个专家（系数=1，纯 delta），用于"专家=域"上限实验
                topks = torch.full((B, self.topk), int(fe), dtype=torch.long, device=prob.device)
                vals = biased.gather(1, topks)
            elif getattr(self, "force_random", False):
                topks = torch.randint(0, self.num_expert, (B, self.topk), device=prob.device)
                vals = biased.gather(1, topks)
            else:
                vals, topks = torch.topk(biased, self.topk, dim=-1)
            # [FUSE_MODE_PATCH] 只改"融合时用哪些专家 + 系数是什么"，不改 self.topk 的口径
            #   sum  ⇒ c_e≡1      ⇒ 融合 = Σ_e ΔW_e（官方"全专家"语义；前向与选了谁无关）
            #   prob ⇒ c_e=prob_e ⇒ 融合 = Σ_e prob_e·ΔW_e（概率加权混合）
            #   mean ⇒ c_e=1/E    ⇒ 专家【等权平均】（配 MOE_NO_SHARED=1 的"完整权重专家"语义）
            _fm = getattr(self, "fuse_mode", "topk")
            if _fm in ("sum", "prob", "mean"):
                topks = torch.arange(self.num_expert, device=prob.device).unsqueeze(0).expand(B, -1).contiguous()
                vals = prob.gather(1, topks)
                _wbp = (_fm == "prob")
            else:
                _wbp = self.weight_by_prob
            # [FUSE_TOPK_PATCH] 融合侧 TopK（默认 0 ⇒ 不启用；_fk=1 等价 topk、_fk=E 等价 prob）
            _fk = int(getattr(self, "fuse_topk", 0) or 0)
            _is_mean = (_fm == "mean")
            if 0 < _fk <= self.num_expert:
                topks = torch.topk(biased, _fk, dim=-1)[1]
                vals = prob.gather(1, topks)
                _wbp = True
                _is_mean = False
            coeff = torch.zeros((B, self.num_expert), device=prob.device)
            if fe is not None:
                _cval = torch.ones_like(vals)
            elif _is_mean:
                _cval = torch.full_like(vals, 1.0 / self.num_expert)
            elif _wbp:
                _cval = vals / vals.detach().sum(-1, keepdim=True)
            else:
                _cval = vals / vals.detach()
            coeff.scatter_(1, topks, _cval)
            # [FUSE_SCALE_PATCH] 融合 delta 的整体幅度缩放（默认 1.0 ⇒ 不改变任何历史行为）
            _fs = float(getattr(self, "fuse_scale", 1.0))
            if _fs != 1.0:
                coeff = coeff * _fs
        else:
            prob = F.softmax(self.router(flat).mean(0), dim=-1)
            biased = prob - self.penalty
            vals, topks = torch.topk(biased, self.topk)
            coeff = torch.zeros(self.num_expert, device=prob.device)
            if self.weight_by_prob:
                coeff.scatter_(0, topks, vals / vals.detach().sum())
            else:
                coeff.scatter_(0, topks, vals / vals.detach())
            coeff = coeff.unsqueeze(0).repeat(B, 1)
            topks = topks.unsqueeze(0).repeat(B, 1)
        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()
        # [V1-lbwindow] batch 级 LB：load 用窗口内累积统计（detach），prob 用当前样本（可导）
        #   window=1 时 load == 本样本的 one-hot，逐位等价于原实现
        self._lb_cnt = cnt if self._lb_cnt is None else self._lb_cnt + cnt
        self._lb_n += 1
        _load = (self._lb_cnt / self._lb_cnt.sum().clamp(min=1)).detach()
        self.lb_loss = self.num_expert * (
            _load * prob.reshape(-1, self.num_expert).mean(0)).sum()
        if self.lb_window > 1 and self._lb_n >= self.lb_window:
            self._lb_cnt = None
            self._lb_n = 0
        self.update_coeff(coeff)
        self.topks = topks
        if self.record_route:
            self.route_prob_list.append(prob.detach().cpu())

    def _get_topks_block(self, x: torch.Tensor):
        """已废弃入口：直接委派给逐样本路径（保留函数名，避免外部引用报错）。"""
        return self.get_topks(x)

    def update_batch_stats(self):
        """[BR] 把本样本累积的统计量并入 batch 级 EMA（策略在每个样本适应结束时调用）"""
        if not getattr(self, "batch_route", False) or self._br_acc is None:
            return
        cur = self._br_acc / max(1, self._br_n)
        self._br_ema = cur if self._br_ema is None else (
            (1.0 - self.br_momentum) * self._br_ema + self.br_momentum * cur)
        self._br_acc = None
        self._br_n = 0

    def reset_batch_stats(self):
        """[BR] 清空 batch 级 EMA（每条流开始时调用）"""
        self._br_ema = None
        self._br_acc = None
        self._br_n = 0

    def step_once(self):
        self.step += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.router is not None:
            self.get_topks(x)
        assert (self.coeff is not None) and (self.topks is not None), "no routing info"

        _add_sh = not getattr(self, "no_shared", False)      # [EXPERT_INIT_PATCH]
        if self.coeff.dim() == 3:
            # 逐帧/外部 3D coeff: (B,T,E)/(B,T,K)
            final_weight, final_bias = fuse_params(
                self.experts_weight, self.experts_bias,
                self.topks, self.coeff,
                self.weight, self.bias, _add_sh)
        elif self.coeff.dim() == 2:
            # 逐样本 2D coeff: (B,E)/(B,K) —— 方案 A 主路径
            final_weight, final_bias = fuse_params_samplewise(
                self.experts_weight, self.experts_bias,
                self.topks, self.coeff,
                self.weight, self.bias, _add_sh)
        else:
            raise ValueError(f"unexpected coeff.dim={self.coeff.dim()}")

        if isinstance(self.base_mod, nn.LayerNorm):
            result = apply_layernorm(x, final_weight, final_bias, self.base_mod.eps)
            self.cls_feature = result[:, 0]
            self.shared_feature = F.layer_norm(
                x[:, 0],
                (x.shape[-1],),
                weight=self.weight,
                bias=self.bias,
                eps=self.base_mod.eps,
            )
            return result
        elif isinstance(self.base_mod, nn.BatchNorm2d):
            return apply_batchnorm(x, final_weight, final_bias, self.base_mod.eps)
        elif isinstance(self.base_mod, nn.GroupNorm):
            return apply_groupnorm(
                x, self.base_mod.num_groups, final_weight, final_bias, self.base_mod.eps
            )
        else:
            raise NotImplementedError(f"Unsupported norm module: {type(self.base_mod)}")