import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple


@torch.jit.script
def fuse_linear_params(
    experts_weight: torch.Tensor,
    experts_bias: torch.Tensor,
    topks: torch.Tensor,
    coeff: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_bias: torch.Tensor,
    has_bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    针对 Linear 层的参数融合
    experts_weight: [E, Out, In]
    experts_bias: [E, Out]
    topks: [B, K]
    coeff: [B, E]
    shared_weight: [Out, In]
    shared_bias: [Out]
    """
    BT, K = topks.shape
    E, Out, In = experts_weight.shape

    # 1. 提取被选中的 top-k 专家的权重系数
    selected_coeff = coeff.gather(1, topks)  # [BT, K]

    # 2. 聚集选中的专家权重
    # experts_weight 扩展为 [BT, E, Out, In]
    expanded_weights = experts_weight.unsqueeze(0).expand(BT, -1, -1, -1)
    expanded_topks = topks.view(BT, K, 1, 1).expand(BT, K, Out, In)
    selected_weight = expanded_weights.gather(1, expanded_topks)  # [BT, K, Out, In]

    # 3. 按系数进行加权求和
    selected_coeff_w = selected_coeff.view(BT, K, 1, 1)
    fused_weight = (selected_weight * selected_coeff_w).sum(dim=1)  # [BT, Out, In]
    final_weight = fused_weight + shared_weight.unsqueeze(0)        # 加上共享权重

    # 4. 如果有偏置，进行同样的聚集和加权计算
    if has_bias:
        expanded_bias = experts_bias.unsqueeze(0).expand(BT, -1, -1)
        expanded_topks_b = topks.view(BT, K, 1).expand(BT, K, Out)
        selected_bias = expanded_bias.gather(1, expanded_topks_b)   # [BT, K, Out]
        
        selected_coeff_b = selected_coeff.view(BT, K, 1)
        fused_bias = (selected_bias * selected_coeff_b).sum(dim=1)  # [BT, Out]
        final_bias = fused_bias + shared_bias.unsqueeze(0)
    else:
        final_bias = experts_bias # 用作占位，实际不用

    return final_weight, final_bias


@torch.jit.script
def apply_linear(
    x: torch.Tensor,      # [BT, In]
    weight: torch.Tensor, # [BT, Out, In]
    bias: torch.Tensor,   # [BT, Out]
    has_bias: bool
) -> torch.Tensor:
    """
    使用融合后的动态权重进行线性投影 (Batch * Time 平铺版)
    """
    # x: [BT, In] -> [BT, 1, In]
    # weight.transpose: [BT, In, Out]
    # bmm result: [BT, 1, Out] -> [BT, Out]
    out = torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if has_bias:
        out = out + bias
    return out


class MoELinearLayer(nn.Module):
    """
    通用 MoE 线性层（Linear）模块。
    """

    def __init__(
        self,
        idx: int,
        num_expert: int,
        activate_shared_expert: bool,
        base_mod: nn.Linear,
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
        frames_per_sec: int = 50,
    ):
        super().__init__()
        assert num_expert > 0, "专家数必须大于 0"
        assert isinstance(base_mod, nn.Linear), "base_mod 必须是 nn.Linear"
        
        self.idx = idx
        self.num_expert = num_expert
        self.base_mod = base_mod
        self.self_router = self_router
        self.frames_per_sec = frames_per_sec

        # 从传入的 base_mod 提取基础参数 (共享专家)
        self.weight = base_mod.weight  # shape: [out_features, in_features]
        self.weight.requires_grad_(activate_shared_expert)
        
        self.has_bias = base_mod.bias is not None
        if self.has_bias:
            self.bias = base_mod.bias  # shape: [out_features]
            self.bias.requires_grad_(activate_shared_expert)
        else:
            # JIT Script 要求类型一致，若原层无bias则创建一个空的作占位
            self.bias = nn.Parameter(torch.empty(0, device=device), requires_grad=False)

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
            self.experts_bias = nn.Parameter(torch.empty(0, device=device), requires_grad=False)

        # 缩放专家参数 (让初始时的 norm 与原始网络权重成比例)
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

        # 路由器
        if self_router:
            in_features = self.weight.size(1)
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
        self.pass_through_coeff = pass_through_coeff
        self.cnt = torch.zeros(num_expert, device=device)
        self.lb_loss = None
        self.route_prob_list = []
        self.record_route = False


        if grad_hook:
            hook_fn = self.make_step_aware_hook(f"linear_layer{self.idx}", self)
            self.register_expert_block_hook(
                self.experts_weight, kind="weight", idx=self.idx, hook_fn=hook_fn
            )
            if self.has_bias:
                self.register_expert_block_hook(
                    self.experts_bias, kind="bias", idx=self.idx, hook_fn=hook_fn
                )
            if hasattr(self, "router") and self.router is not None:
                self.register_router_separate_hooks(
                    self.router, f"linear_layer{self.idx}/router", hook_fn
                )

    # ... (此处省略 make_step_aware_hook, register_expert_block_hook, 
    # register_router_separate_hooks, get_trainable_params, update_coeff 代码，与你提供的原代码完全相同) ...
    
    @staticmethod
    def make_step_aware_hook(name_prefix, instance):
        def hook_fn(name, norm_tensor):
            full_name = f"{name_prefix}/{name}"
        return hook_fn

    @staticmethod
    def register_expert_block_hook(param, kind: str, idx: int, hook_fn: Callable):
        def hook(grad):
            grad_flat = grad.view(grad.size(0), -1)  # [num_expert, D]
            mask = grad_flat.abs().sum(dim=1) > 0 
            active_grads = grad_flat[mask]
            if active_grads.numel() > 0:
                norms = active_grads.norm(p=2, dim=1) 
                hook_fn(f"layer{idx}/expert_{kind}_grad_max", norms.max())
                hook_fn(f"layer{idx}/expert_{kind}_grad_mean", norms.mean())
            return grad
        param.register_hook(hook)

    @staticmethod
    def register_router_separate_hooks(router: nn.Linear, name_prefix: str, hook_fn: Callable):
        router.weight.register_hook(lambda grad: hook_fn(f"{name_prefix}_weight_grad_norm", grad.norm(2)))
        router.bias.register_hook(lambda grad: hook_fn(f"{name_prefix}_bias_grad_norm", grad.norm(2)))

    def update_coeff(self, coeff: torch.Tensor):
        self.coeff = coeff if self.pass_through_coeff else coeff.detach()

    def get_trainable_params(self):
        params = [self.experts_weight]
        if self.has_bias:
            params.append(self.experts_bias)
        params.append(self.weight)
        if self.has_bias:
            params.append(self.bias)
        if hasattr(self, "router"):
            for name, p in self.router.named_parameters():
                params.append(p)
        return params

    def get_topks(self, x: torch.Tensor):
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

    def step_once(self):
        self.step += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (hasattr(self, 'router') and self.router is not None) or (self.coeff is not None), "需要路由器或融合系数"
        
        if hasattr(self, 'router') and self.router is not None:
            self.get_topks(x)

        is_3d = (x.dim() == 3)
        B = x.size(0)
        
        # 将输入展平为 [B*T, ...] 以匹配 JIT
        if is_3d:
            T = x.size(1)
            x_flat = x.reshape(B * T, -1)
            coeff_flat = self.coeff.reshape(B * T, -1)
            topks_flat = self.topks.reshape(B * T, -1)
        else:
            T = 1
            x_flat = x
            coeff_flat = self.coeff
            topks_flat = self.topks
            
        # BUG-010 fix: 不物化 [B*T, Out, In] 逐帧融合权重（O(BT*D^2) 显存爆炸）。
        # 数学等价式: y = x@W_shared^T + b_shared + Σ_e c_{.,e} * (x@E_e^T + b_e)
        # （coeff 对未选中专家为 0，故 Σ_e 实际只在 top-k 上生效；内存降为 O(BT*D)）
        bias = self.bias if self.has_bias else None
        out = F.linear(x_flat, self.weight, bias)
        e_bias = self.experts_bias if self.has_bias else None
        for e in range(self.num_expert):
            c_e = coeff_flat[:, e].unsqueeze(-1)          # (BT, 1)
            eb = None if e_bias is None else e_bias[e]
            delta_e = F.linear(x_flat, self.experts_weight[e], eb)
            out = out + c_e * delta_e

        if is_3d:
            return out.reshape(B, T, -1)
        return out