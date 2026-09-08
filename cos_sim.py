import torch
import math
from transformers import Wav2Vec2ForCTC
import torch.nn as nn

def calculate_adaptation_cosine_similarity(
    pretrained_model_name,
    model1_path,
    model2_path,
    threshold=1.1
):
    print("1. 正在加载预训练模型...")
    model_pre = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name)

    print("2. 正在加载 Adapt 后的模型权重...")
    state_dict1 = torch.load(model1_path, map_location="cpu")
    state_dict2 = torch.load(model2_path, map_location="cpu")

    # 兼容：如果保存的是整个模型
    if hasattr(state_dict1, "state_dict"):
        state_dict1 = state_dict1.state_dict()
    if hasattr(state_dict2, "state_dict"):
        state_dict2 = state_dict2.state_dict()

    print("3. 正在按层收集参数并计算余弦相似度...")
    print(f"--- 以下是余弦相似度低于 {threshold} 的层 ---")

    layer_vectors = {}

    # ===============================
    # Step 1: 按层收集向量
    # ===============================
    for name, param_pre in model_pre.named_parameters():

        module_name = name.rsplit(".", 1)[0]
        module = dict(model_pre.named_modules()).get(module_name, None)

        is_feature = (
            "feature_extractor" in name
            or "feature_projection" in name
        )

        is_ln = isinstance(module, torch.nn.LayerNorm)

        # 只分析 feature_extractor / feature_projection / LayerNorm
        if not (is_feature or is_ln):
            continue

        if name not in state_dict1 or name not in state_dict2:
            print(f"警告: 参数 {name} 在微调模型中未找到，已跳过。")
            continue

        p_pre = param_pre.data.cpu().view(-1)
        p1 = state_dict1[name].cpu().view(-1)
        p2 = state_dict2[name].cpu().view(-1)

        if p_pre.shape != p1.shape or p_pre.shape != p2.shape:
            print(f"跳过 {name}: 形状不匹配")
            continue

        v1 = p1 - p_pre
        v2 = p2 - p_pre

        if module_name not in layer_vectors:
            layer_vectors[module_name] = {"v1": [], "v2": []}

        layer_vectors[module_name]["v1"].append(v1)
        layer_vectors[module_name]["v2"].append(v2)

    # ===============================
    # Step 2: 按层计算余弦相似度
    # ===============================
    layer_similarities = {}

    global_dot_product = 0.0
    global_norm1_sq = 0.0
    global_norm2_sq = 0.0

    for layer_name, vecs in layer_vectors.items():

        v1 = torch.cat(vecs["v1"])
        v2 = torch.cat(vecs["v2"])

        dot = torch.dot(v1, v2).item()
        norm1_sq = torch.dot(v1, v1).item()
        norm2_sq = torch.dot(v2, v2).item()

        norm1 = math.sqrt(norm1_sq)
        norm2 = math.sqrt(norm2_sq)

        if norm1 == 0.0 or norm2 == 0.0:
            continue

        cos_sim = dot / (norm1 * norm2)

        layer_similarities[layer_name] = cos_sim

        if cos_sim < threshold:
            print(f"层名称: {layer_name:<60} | 相似度: {cos_sim:.4f}")

        # 累加用于全局计算
        global_dot_product += dot
        global_norm1_sq += norm1_sq
        global_norm2_sq += norm2_sq

    print("-" * 50)

    # ===============================
    # Step 3: 计算全局余弦相似度
    # ===============================
    global_norm1 = math.sqrt(global_norm1_sq)
    global_norm2 = math.sqrt(global_norm2_sq)

    if global_norm1 == 0.0 or global_norm2 == 0.0:
        global_cos_sim = 0.0
    else:
        global_cos_sim = global_dot_product / (global_norm1 * global_norm2)

    print(f"计算完成！全局余弦相似度为: {global_cos_sim:.4f}")

    return layer_similarities, global_cos_sim

if __name__ == "__main__":
    # 替换为你使用的预训练模型名称或本地路径
    PRETRAINED_MODEL = "/root/autodl-tmp/model/wav2vec2" 
    '''MODEL1_PATH = "/root/autodl-tmp/model/CSUTA/AA/model.pt"
    MODEL2_PATH = "/root/autodl-tmp/model/CSUTA/AC/model.pt"

    layer_similarities, global_cos_sim = calculate_adaptation_cosine_similarity(PRETRAINED_MODEL, MODEL1_PATH, MODEL2_PATH)'''

    model_pre = Wav2Vec2ForCTC.from_pretrained(PRETRAINED_MODEL)
    modules_dict = dict(model_pre.named_modules())
    
    for name, param in model_pre.named_parameters():
        if param.requires_grad:
            # 去掉参数名最后的 weight / bias，得到模块路径
            module_name = name.rsplit('.', 1)[0]
    
            try:
                module = modules_dict[module_name]
                module_type = type(module).__name__
            except KeyError:
                module = None
                module_type = "Unknown"
    
            '''print(f"{name}")
            print(f"  ├─ shape: {tuple(param.shape)}")
            print(f"  ├─ numel: {param.numel()}")'''
            
            # 判断是否为卷积层 (Wav2Vec2 主要使用 Conv1d，这里同时兼容 2d 和 3d)
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                print(f"  ├─ module type: {module_type}")
                print(f"  ├─ in_channels: {module.in_channels}")
                print(f"  ├─ out_channels: {module.out_channels}")
                print(f"  ├─ kernel_size: {module.kernel_size}")
                print(f"  ├─ stride: {module.stride}")
                print(f"  ├─ padding: {module.padding}")
                print(f"  ├─ dilation: {module.dilation}")
                print(f"  └─ groups: {module.groups}\n")
            else:
                print(f"  └─ module type: {module_type}\n")