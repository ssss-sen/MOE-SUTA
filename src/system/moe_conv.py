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
    has_bias: bool
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
    selected_coeff = coeff.unsqueeze(-1)
    fused_weight = (selected_weight * selected_coeff).sum(dim=1)
    
    # 加上共享权重并还原形状
    final_weight = fused_weight + shared_weight.view(-1).unsqueeze(0)
    final_weight = final_weight.view(B, *w_shape)

    # 2. 融合偏置 (如果有)
    if has_bias:
        b_shape = shared_bias.shape
        eb_flat = experts_bias.view(E, -1)
        expanded_topks_b = topks.unsqueeze(-1).expand(-1, -1, eb_flat.size(1))
        
        selected_bias = eb_flat.unsqueeze(0).expand(B, -1, -1).gather(1, expanded_topks_b)
        fused_bias = (selected_bias * selected_coeff).sum(dim=1)
        
        final_bias = fused_bias + shared_bias.view(-1).unsqueeze(0)
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
        """
        按帧率切块（Block-wise）计算路由分布。
        x shape: [B, in_C, L]
        返回: 动态切分的 block 数量
        """
        B, in_C, L = x.shape
        # 根据本层一秒的时长推断需要分为多少个Block
        fps = max(1, int(self.frames_per_sec))
        num_blocks = math.ceil(L / fps)

        # 将序列补齐到帧率的整数倍，方便路由池化
        pad_len = (fps - (L % fps)) % fps
        x_pad_route = F.pad(x, (0, pad_len))
        
        # 重塑并对每个 Block 在时间维度上求均值: [B, in_C, num_blocks, F] -> [B, num_blocks, in_C]
        x_blocks = x_pad_route.view(B, in_C, num_blocks, fps)
        block_feats = x_blocks.mean(dim=-1).transpose(1, 2)
        
        # 展平为 [B*num_blocks, in_C] 进行路由预测
        flat = block_feats.reshape(B * num_blocks, in_C)
        prob = F.softmax(self.router(flat), dim=-1)
        
        biased = prob - self.penalty
        vals, topks = torch.topk(biased, self.topk, dim=-1)
        
        coeff = torch.zeros_like(prob)
        if self.weight_by_prob:
            coeff.scatter_(1, topks, vals / vals.detach().sum(-1, keepdim=True))
        else:
            coeff.scatter_(1, topks, vals / vals.detach())
            
        # 统计平衡
        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()
        
        # 计算 LB Loss (使用平均重要性)
        importance = prob.mean(dim=0)
        self.lb_loss = self.num_expert * torch.sum(importance * (cnt / cnt.sum()))
        
        self.update_coeff(coeff)
        self.topks = topks
        if self.record_route:
            self.route_prob_list.append(prob.detach().cpu())
        
        return num_blocks

    def step_once(self):
        self.step += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, in_C, L = x.shape
        
        if self.router is not None:
            num_blocks = self.get_topks(x)
        else:
            num_blocks = math.ceil(L / max(1, self.frames_per_sec))
            
        # topks 和 coeff 现在的形状都是 [B * num_blocks, K 或 E]
        # 融合出每一块专属于该时段的权重: shape = [B * num_blocks, out_C, in_C/g, K]
        final_weight, final_bias = fuse_conv_params(
            self.experts_weight, self.experts_bias, self.topks,
            self.coeff, self.weight, self.bias, self.has_bias
        )

        # =============== 无边界伪影的 Block-wise 动态卷积 ===============
        P = self.padding
        D = self.dilation
        S = self.stride
        K_size = final_weight.size(-1)
        K_eff = (K_size - 1) * D + 1
        
        # 1. 计算理论上正常 Conv1d 应该输出的长度
        L_out_total = (L + 2 * P - K_eff) // S + 1
        
        # 2. 将输出均匀切分为 num_blocks 个块
        F_out_chunk = math.ceil(L_out_total / num_blocks)
        
        # 3. 逆向计算每个输出块所需的确切输入窗口大小 (涵盖必要的重叠上下文)
        L_in_chunk = (F_out_chunk - 1) * S + K_eff
        
        # 4. 像常规 Conv1d 一样在两端做 Padding
        x_pad = F.pad(x, (P, P))
        
        # 5. 为了防止最后一个滑动窗口超出边界，向右补充足够的长度
        required_len = (num_blocks - 1) * (F_out_chunk * S) + L_in_chunk
        if x_pad.size(-1) < required_len:
            x_pad = F.pad(x_pad, (0, required_len - x_pad.size(-1)))
            
        # 6. 利用 unfold 提取滑动窗口，提取出包含重叠信息的完美 Block 序列
        # 输出: [B, in_C, num_blocks, L_in_chunk]
        x_unfold = x_pad.unfold(dimension=-1, size=L_in_chunk, step=F_out_chunk * S)
        
        # 7. 转置并重塑，把 Batch 维度和 Block 维度合并 -> [B * num_blocks, in_C, L_in_chunk]
        x_unfold = x_unfold.transpose(1, 2).contiguous().view(B * num_blocks, in_C, L_in_chunk)
        
        # 8. 执行分组动态卷积（此时 Padding=0，因为 x_unfold 已经包含填充上下文）
        out_chunks = apply_dynamic_conv1d(
            x_unfold, final_weight, final_bias, 
            S, 0, D, self.groups, self.has_bias
        ) # 输出形状: [B * num_blocks, out_C, F_out_chunk]
        
        # 9. 还原维度并拼接 -> [B, out_C, num_blocks * F_out_chunk]
        out_C = out_chunks.size(1)
        out = out_chunks.view(B, num_blocks, out_C, F_out_chunk)
        out = out.transpose(1, 2).reshape(B, out_C, num_blocks * F_out_chunk)
        
        # 10. 切去多余部分，使长度与标准卷积输出百分之百对齐
        return out[:, :, :L_out_total]