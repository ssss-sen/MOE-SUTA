import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional


# @torch.jit.script
def fuse_conv_params(
    experts_weight: torch.Tensor,
    experts_bias: torch.Tensor,
    topks: torch.Tensor,
    coeff: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_bias: torch.Tensor,
    has_bias: bool,
    add_shared: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    针对 Conv 权重的融合逻辑。
    为了适应任意维度的权重，将其在专家维度外进行展平 (Flatten)，融合后再 reshape 回来。
    topks 和 coeff 的形状假设为 (B, K) (Sample-wise routing)。
    """
    B, K = topks.shape
    E = experts_weight.size(0)
    w_shape = shared_weight.shape

    # 1. 融合权重
    # [E, out_C, in_C, K] -> [E, D]
    ew_flat = experts_weight.view(E, -1)
    
    # [B, K, D]
    expanded_topks_w = topks.unsqueeze(-1).expand(-1, -1, ew_flat.size(1))
    
    # gather: 选出对应的专家权重并乘以系数
    selected_weight = ew_flat.unsqueeze(0).expand(B, -1, -1).gather(1, expanded_topks_w)
    # [FUSE_TOPK_PATCH] K>1 先按 topks gather（旧写法只对 K=1 成立；K=1 数学等价 ⇒ 默认不变）
    selected_coeff = coeff.gather(1, topks).unsqueeze(-1)
    fused_weight = (selected_weight * selected_coeff).sum(dim=1)
    
    # 加上共享权重并还原形状（[EXPERT_INIT_PATCH] add_shared=False ⇒ 专家替换基座）
    if add_shared:
        final_weight = fused_weight + shared_weight.view(-1).unsqueeze(0)
    else:
        final_weight = fused_weight
    final_weight = final_weight.view(B, *w_shape)

    # 2. 融合偏置 (如果有)
    if has_bias:
        b_shape = shared_bias.shape
        eb_flat = experts_bias.view(E, -1)
        expanded_topks_b = topks.unsqueeze(-1).expand(-1, -1, eb_flat.size(1))
        
        selected_bias = eb_flat.unsqueeze(0).expand(B, -1, -1).gather(1, expanded_topks_b)
        fused_bias = (selected_bias * selected_coeff).sum(dim=1)
        
        if add_shared:
            final_bias = fused_bias + shared_bias.view(-1).unsqueeze(0)
        else:
            final_bias = fused_bias
        final_bias = final_bias.view(B, *b_shape)
    else:
        # TorchScript 要求返回相同类型的张量，所以返回一个空的张量
        final_bias = torch.empty(0, device=shared_weight.device)

    return final_weight, final_bias


def apply_dynamic_conv1d(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor, 
    stride: int,
    padding: int, 
    dilation: int, 
    groups: int,
    has_bias: bool
) -> torch.Tensor:
    """
    利用 Group Convolution 技巧实现 Batch 级别的动态卷积。
    x shape: [B, in_C, L]
    weight shape: [B, out_C, in_C/groups, K]
    bias shape: [B, out_C]
    """
    B, in_C, L = x.shape
    _, out_C, in_C_per_group, K = weight.shape

    # 将 Batch 维度融入 Channel 维度：[1, B * in_C, L]
    x_reshaped = x.view(1, B * in_C, L)
    
    # 将 Batch 维度融入 Output Channel 维度: [B * out_C, in_C/groups, K]
    weight_reshaped = weight.view(B * out_C, in_C_per_group, K)
    
    if has_bias:
        bias_reshaped = bias.view(B * out_C)
    else:
        bias_reshaped = None

    # 分组数需要乘以 B，使得每个样本独立与自己的权重做卷积
    out = F.conv1d(
        x_reshaped,
        weight=weight_reshaped,
        bias=bias_reshaped,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=B * groups
    )

    # 恢复 Batch 维度: [B, out_C, L_out]
    out = out.view(B, out_C, out.size(-1))
    return out


class MoEConv1dLayer(nn.Module):
    """
    Conv1d MoE 层实现。
    将原始 Conv1d 转换为具有多个并行专家权重的动态路由卷积层。
    """

    def __init__(
        self,
        idx: int,
        num_expert: int,
        activate_shared_expert: bool,
        base_mod: nn.Conv1d,
        randomness: float,
        self_router: bool = False,
        samplewise: bool = None,
        topk: int = None,
        weight_by_prob: bool = None,
        penalty: float = None,
        decay: float = None,
        logger: Callable = None,
        grad_hook: bool = False,
        pass_through_coeff: bool = True,
        device: str = "cuda",
        frames_per_sec: int = 16000,
    ):
        super().__init__()
        assert num_expert > 0, "专家数必须大于 0"
        self.idx = idx
        self.num_expert = num_expert
        self.base_mod = base_mod
        self.self_router = self_router
        self.frames_per_sec = frames_per_sec

        # Conv1d 的特定参数
        self.stride = base_mod.stride[0] if isinstance(base_mod.stride, tuple) else base_mod.stride
        self.padding = base_mod.padding[0] if isinstance(base_mod.padding, tuple) else base_mod.padding
        self.dilation = base_mod.dilation[0] if isinstance(base_mod.dilation, tuple) else base_mod.dilation
        self.groups = base_mod.groups
        self.has_bias = base_mod.bias is not None

        # 提取基础参数 (共享专家)
        self.weight = base_mod.weight
        self.weight.requires_grad_(activate_shared_expert)
        if self.has_bias:
            self.bias = base_mod.bias
            self.bias.requires_grad_(activate_shared_expert)
        else:
            self.register_buffer("bias", torch.empty(0, device=device))

        # 初始化专家参数
        weight_shape = self.weight.shape
        self.experts_weight = nn.Parameter(
            torch.randn((num_expert,) + tuple(weight_shape), device=device)
        )
        
        if self.has_bias:
            bias_shape = self.bias.shape
            self.experts_bias = nn.Parameter(
                torch.randn((num_expert,) + tuple(bias_shape), device=device)
            )
        else:
            self.register_buffer("experts_bias", torch.empty(0, device=device))

        # 缩放专家参数
        w_norm = torch.norm(self.weight).item()
        expert_w_norm = torch.norm(self.experts_weight.view(num_expert, -1), dim=1)
        factor_w = (randomness * w_norm / expert_w_norm).view(
            num_expert, *([1] * len(weight_shape))
        )
        self.experts_weight.data *= factor_w

        if self.has_bias:
            b_norm = torch.norm(self.bias).item()
            expert_b_norm = torch.norm(self.experts_bias.view(num_expert, -1), dim=1)
            factor_b = (randomness * b_norm / expert_b_norm).view(
                num_expert, *([1] * len(bias_shape))
            )
            self.experts_bias.data *= factor_b

        # 路由器 - Conv1d 特征提取一般按通道计算(in_channels)
        if self_router:
            in_features = base_mod.in_channels
            self.router = nn.Linear(in_features, num_expert, device=device)
            nn.init.xavier_normal_(self.router.weight)
            nn.init.zeros_(self.router.bias)

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
        self.logger = logger
        self.pass_through_coeff = pass_through_coeff
        self.cnt = torch.zeros(num_expert, device=device)
        self.lb_loss = None
        self.route_prob_list = []
        self.record_route = False
        self.force_random = False
        # [BR] batch 级路由：用"跨样本统计量"(EMA) 做专家决策（卷积层同样支持）
        self.batch_route = False
        self.br_momentum = 0.5
        self._br_ema = None
        self._br_acc = None
        self._br_n = 0
        # [ORACLE] 强制专家号（None = 正常路由）
        self.force_expert = None
        # [RG] 路由粒度："native"(逐块) / "sample"(整条样本一个决策)
        self.route_granularity = "native"



    def get_trainable_params(self):
        params = [self.experts_weight, self.weight]
        if self.has_bias:
            params.extend([self.experts_bias, self.bias])
        if hasattr(self, "router"):
            for name, p in self.router.named_parameters():
                params.append(p)
        return params

    def update_coeff(self, coeff: torch.Tensor):
        self.coeff = coeff if self.pass_through_coeff else coeff.detach()

    def get_topks(self, x: torch.Tensor):
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
        """MoE 卷积：每条样本一次专家决策，融合一次权重后做一次分组卷积。"""
        B, in_C, L = x.shape
        if self.router is not None:
            self.get_topks(x)          # 只产生一次决策：self.topks [B,K] / self.coeff [B,E]
        assert self.topks is not None and self.coeff is not None, \
            "[SAMPLE_LEVEL] 需要路由器，或由外部注入的 coeff/topks"
        assert self.topks.dim() == 2 and self.coeff.dim() == 2, \
            "[SAMPLE_LEVEL] 只支持样本级 2D 路由（[B,K]/[B,E]）"

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
