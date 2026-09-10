import math
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
                    num_expert=self.num_expert,
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
                    num_expert=self.num_expert,
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

    def get_coeff_topks(self, embedding):
        """
        Frame-wise routing for wav2vec.
        embedding: [B, T, C]
        """

        B, T, C = embedding.shape
        F = self.frames_per_sec

        # 按照 F 帧大小对 T 维度进行切片，最后一个 chunk 可能小于 F
        chunks = torch.split(embedding, F, dim=1) 
        # 对每一个 chunk 在时间维度求均值，得到 [B, C] 的列表
        block_feats = [chunk.mean(dim=1) for chunk in chunks] 
        # 堆叠成 Block 级特征，形状为 [B, num_blocks, C]
        block_feat = torch.stack(block_feats, dim=1) 
        num_blocks = block_feat.shape[1]

        block_feat_flat = block_feat.view(B * num_blocks, C)  # [B*num_blocks, C]

        route_prob = F.softmax(self.router(block_feat_flat), dim=-1)  # [B*num_blocks, E]
        biased_prob = route_prob - self.penalty  # broadcast

        prob, topks = torch.topk(biased_prob, self.topk, dim=-1)  # [B*num_blocks, K]

        # 更新专家计数器和惩罚项 (按照 Block 统计)
        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt

        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()

        coeff = torch.zeros_like(route_prob)  # [B*num_blocks, E]

        if self.weight_by_prob:
            norm_prob = prob / prob.detach().sum(-1, keepdim=True)
        else:
            norm_prob = prob / prob.detach()

        coeff.scatter_(1, topks, norm_prob)

        # 恢复 Block 维度: [B, num_blocks, ...]
        coeff = coeff.view(B, num_blocks, self.num_expert)
        topks = topks.view(B, num_blocks, self.topk)
        route_prob = route_prob.view(B, num_blocks, self.num_expert)
        biased_prob = biased_prob.view(B, num_blocks, self.num_expert)

        # 拓展方法：在第2维扩展出 F 个重复值 -> [B, num_blocks, F, ...] -> 展平后截断到实际帧数 T
        coeff_frame = coeff.unsqueeze(2).expand(-1, -1, F, -1).reshape(B, num_blocks * F, self.num_expert)
        coeff_frame = coeff_frame[:, :T, :]  # [B, T, E]

        topks_frame = topks.unsqueeze(2).expand(-1, -1, F, -1).reshape(B, num_blocks * F, self.topk)
        topks_frame = topks_frame[:, :T, :]  # [B, T, K]

        # route_prob 和 biased_prob 按照原代码逻辑输出为平铺格式 [B*T, E] 以兼容外部 hook
        route_prob_frame = route_prob.unsqueeze(2).expand(-1, -1, F, -1).reshape(B, num_blocks * F, self.num_expert)
        route_prob_frame = route_prob_frame[:, :T, :].reshape(B * T, self.num_expert)  # [B*T, E]

        biased_prob_frame = biased_prob.unsqueeze(2).expand(-1, -1, F, -1).reshape(B, num_blocks * F, self.num_expert)
        biased_prob_frame = biased_prob_frame[:, :T, :].reshape(B * T, self.num_expert)  # [B*T, E]

        # 计算 Load Balancing Loss (基于 Block 级别的统计)
        importance = route_prob.view(-1, self.num_expert).mean(dim=0)
        load = cnt / cnt.sum()
        self.lb_loss = self.num_expert * torch.sum(importance * load)

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
            self.current_coeff = 1 / (torch.exp(entropy.detach() - self.e_margin_coeff * math.log(1000)))
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
            loss.append(mod.lb_loss)
        if len(loss) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(loss)

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
        return params