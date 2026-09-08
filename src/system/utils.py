import numpy as np
from pathlib import Path
import torch
import yaml
import random
import functools
import time
from pprint import pformat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def recursive_getattr(obj, attr_path):
    """辅助函数：递归获取属性对象"""
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def recursive_setattr(obj, attr_path, value):
    """
    递归设置属性
    :param obj: 目标对象
    :param attr_path: 属性路径字符串，如 "env.project"
    :param value: 要设置的值
    """
    pre, _, post = attr_path.rpartition(".")
    # 如果有路径（如 a.b.c 中的 a.b），先递归获取 a.b 对象
    target = recursive_getattr(obj, pre) if pre else obj
    setattr(target, post, value)


def count_correct(output, target, topk=(1,)):
    """
    Computes the number of correct predictions for the specified values of topk.

    Args:
        output (Tensor): Model outputs, shape [batch_size, num_classes]
        target (Tensor): Ground truth labels, shape [batch_size]
        topk (tuple): Tuple of integers specifying which top-k accuracies to compute.

    Returns:
        A list with the number of correct predictions for each k in topk.
    """
    maxk = max(topk)

    # topk: get top maxk predicted classes for each sample
    _, indices = output.topk(
        maxk, dim=1, largest=True, sorted=True
    )  # [batch_size, maxk]
    indices = indices.t()  # [maxk, batch_size]
    correct = indices.eq(target.view(1, -1).expand_as(indices))  # [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # count of correct top-k
        res.append(int(correct_k.item()))  # return as int
    return res

def set_nested_attr(root, name: str, value):
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def cosine_similarity(tensor_list):
    """
    计算 list 中所有 1D tensor 之间的 cosine similarity

    参数:
        tensor_list (list of torch.Tensor): 形状相同的 1D tensor 列表

    返回:
        torch.Tensor: (N, N) 形状的对称矩阵，表示两两之间的 cosine similarity
    """
    # 将 list 转换为 2D 矩阵 (N, D)，N 是 tensor 数量，D 是每个 tensor 的长度
    stacked_tensors = torch.stack(tensor_list)  # (N, D)

    # 计算余弦相似度 (cosine similarity) 矩阵
    similarity_matrix = torch.nn.functional.cosine_similarity(
        stacked_tensors.unsqueeze(1),  # (N, 1, D)
        stacked_tensors.unsqueeze(0),  # (1, N, D)
        dim=2,  # 在特征维度计算
    )  # 得到 (N, N) 形状的相似度矩阵

    return similarity_matrix


def symmetric_kl_divergence(mean_list, var_list):
    """
    计算对称 KL 散度矩阵，并归一化

    参数:
        mean_list (list of torch.Tensor): 每个元素是 1D tensor，表示正态分布的均值 (mu)
        var_list (list of torch.Tensor): 每个元素是 1D tensor，表示正态分布的方差 (sigma^2)

    返回:
        torch.Tensor: (N, N) 形状的对称 KL 散度矩阵（已归一化）
    """
    # 转换为张量 (N, D)
    means = torch.stack(mean_list)  # (N, D)
    vars = torch.stack(var_list)  # (N, D)

    N, D = means.shape  # N 是分布个数，D 是向量长度

    # 扩展维度进行广播计算 (N, 1, D) 和 (1, N, D) 变为 (N, N, D)
    mu_p = means.unsqueeze(1)  # (N, 1, D)
    mu_q = means.unsqueeze(0)  # (1, N, D)
    var_p = vars.unsqueeze(1)  # (N, 1, D)
    var_q = vars.unsqueeze(0)  # (1, N, D)

    # 计算 KL(p || q)
    kl_pq = 0.5 * (
        (var_q / var_p) + ((mu_p - mu_q) ** 2) / var_p - 1 + torch.log(var_p / var_q)
    ).sum(dim=2)

    # 计算 KL(q || p)
    kl_qp = 0.5 * (
        (var_p / var_q) + ((mu_q - mu_p) ** 2) / var_q - 1 + torch.log(var_q / var_p)
    ).sum(dim=2)

    # 计算对称 KL 散度
    sym_kl_matrix = 0.5 * (kl_pq + kl_qp)

    # 归一化 (除以 D)
    sym_kl_matrix /= D

    return sym_kl_matrix


def lower_triangle_mean(mat: torch.Tensor) -> torch.Tensor:
    """
    计算一个方阵的下三角区域（不含主对角线）的平均值

    参数:
        mat (torch.Tensor): 一个二维方阵张量，形状为 (n, n)

    返回:
        torch.Tensor: 下三角区域的平均值（标量张量）
    """
    if mat.ndim != 2 or mat.size(0) != mat.size(1):
        raise ValueError("Input must be a 2D square matrix.")

    # 创建下三角 mask（不含主对角线）
    mask = torch.tril(torch.ones_like(mat, dtype=torch.bool), diagonal=-1)

    # 提取下三角元素并求平均
    return mat[mask].mean()


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)