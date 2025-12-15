import dgl
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import networkx as nx
from dgl import DGLGraph
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from torch import cosine_similarity
from typing import Optional
import pandas as pd
import os
from find_node import *



def get_high_degree_threshold(degrees, n_std=2.0):
    """
        基于Z-score的自适应阈值计算
        :param degrees: 节点度张量
        :param n_std: 标准差倍数（默认2σ）
        :return: 高度节点阈值
    """
    mean = degrees.float().mean()
    std = degrees.float().std()
    print('n_std:', n_std)
    print('degree mean and std:', mean, std)
    return mean + n_std * std


def probabilistic_delete(similarity, delete_condition, delete_prob, temperature):
    """
       基于相似度的概率删除函数
       :param similarity: 边的余弦相似度
       :param delete_condition: 候选边的布尔掩码
       :param temperature: 控制概率分布的尖锐程度
       :return: 最终删除的边索引
    """
    candidate_indices = delete_condition.nonzero().squeeze()
    if candidate_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    candidate_sim = similarity[candidate_indices]
    normalized_sim = (candidate_sim - candidate_sim.min()) / (candidate_sim.max() - candidate_sim.min() + 1e-8)
    weights = 1 - torch.sigmoid((normalized_sim - 0.5) / temperature)
    probs = weights / weights.sum()

    sample_size = int(delete_prob * len(candidate_indices))
    sampled_indices = torch.multinomial(probs, sample_size, replacement=False)

    return candidate_indices[sampled_indices]

def filter_edge_bisimilarity_optimized(graph, features, delete_prob=1, temperature=0.5 , n_std=4, delta=0.):
    device = graph.device

    # Step 1: 自适应高入度节点检测
    degrees = graph.in_degrees()
    # threshold = torch.quantile(degrees.float(), 0.8)  # 70%分位
    threshold = get_high_degree_threshold(degrees, n_std=n_std)
    high_degree_nodes = torch.nonzero(degrees > threshold, as_tuple=True)[0]
    print(f"[INFO] 高入度节点数量: {high_degree_nodes.numel()}")

    # Step 2: 边相似性计算
    src, dst = graph.edges()
    similarity = F.cosine_similarity(features[src], features[dst], dim=1)

    # Step 3: 预计算每个高入度节点的邻域相似性均值+标准差
    node_in_avg = torch.zeros(graph.num_nodes(), dtype=torch.float, device=device)
    node_in_std = torch.zeros(graph.num_nodes(), dtype=torch.float, device=device)

    for node in high_degree_nodes:
        preds = graph.predecessors(node)
        if preds.numel() > 0:
            sim_in = F.cosine_similarity(features[node].unsqueeze(0), features[preds], dim=1)
            node_in_avg[node] = sim_in.mean()
            node_in_std[node] = sim_in.std()
            # print(f'[INFO] 节点 {node} 邻域均值: {node_in_avg[node]:.4f}, 标准差: {node_in_std[node]:.4f}')

    # Step 4: 严格双向约束 + 增强版：低于邻域均值 - delta * std才判定异配
    src_high = torch.isin(src, high_degree_nodes)
    dst_high = torch.isin(dst, high_degree_nodes)

    threshold_src = node_in_avg[src] - delta * node_in_std[src]
    threshold_dst = node_in_avg[dst] - delta * node_in_std[dst]

    strict_condition = (
        (((similarity <= threshold_src) & (similarity <= threshold_dst))) & (src_high | dst_high)
    )

    print(f"[DEBUG] strict_condition shape: {strict_condition.shape}")
    print(f"[DEBUG] True count: {strict_condition.sum().item()}/{strict_condition.numel()}")

    # Step 5: 动态概率删边：越异配删概率越高
    delete_indices = probabilistic_delete(
        similarity, strict_condition,
        delete_prob=delete_prob, temperature=temperature
    )

    # if delete_indices.numel() > 0:
    #     deleted_src = src[delete_indices]
    #     deleted_dst = dst[delete_indices]
    #     print(f"[INFO] 删除边数量: {delete_indices.numel()}")
    #     graph = dgl.remove_edges(graph, delete_indices)
    # else:
    #     deleted_src = torch.tensor([], dtype=torch.long, device=device)
    #     deleted_dst = torch.tensor([], dtype=torch.long, device=device)
    deleted_src = src[delete_indices]
    deleted_dst = dst[delete_indices]
    # valid_mask = (degrees[deleted_src] <= threshold) & (degrees[deleted_dst] <= threshold)
    # deleted_src = deleted_src[valid_mask]
    # deleted_dst = deleted_dst[valid_mask]
    #删边
    if deleted_src.numel() > 0:
        print(f"[INFO] 删除边数量: {deleted_src.numel()}")
        delete_eids = graph.edge_ids(deleted_src, deleted_dst)
        graph = dgl.remove_edges(graph, delete_eids)
    #
    #     # 可视化其中一个删边节点的自我网络
    #     # visualize_ego_network(graph, int(deleted_dst[0]), "deleted_node_ego.pdf")
    #     print(f"[INFO] 删除边自我网络已保存为: deleted_node_ego.pdf")
    # else:
    #     deleted_src = torch.tensor([], dtype=torch.long, device=device)
    #     deleted_dst = torch.tensor([], dtype=torch.long, device=device)

    return graph, deleted_src, deleted_dst


def add_edges_to_least_connected(graph):
    # 获取入度
    in_degrees = graph.in_degrees()
    avg_degree = torch.mean(in_degrees.float()).item()
    k = int(avg_degree * 5)
    print('增加邻居数量k：', k)

    # 选择入度最低50%的节点
    sorted_nodes = torch.argsort(in_degrees)
    least_connected = sorted_nodes[:int(len(sorted_nodes) * 0.1)]
    print('需要加边的节点数量:', len(least_connected))

    # 特征归一化
    node_features = graph.ndata['feature'].float()
    norms = torch.norm(node_features, p=2, dim=1, keepdim=True)
    norms[norms == 0] = 1e-6
    normalized_features = node_features / norms

    # 邻接字典
    adjacency_dict = {
        int(n): set(graph.predecessors(n).tolist())
        for n in graph.nodes()
    }

    # 高入度节点定义
    threshold = get_high_degree_threshold(in_degrees, n_std=2)
    high_degree_nodes = torch.nonzero(in_degrees > threshold, as_tuple=True)[0]
    high_degree_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    high_degree_mask[high_degree_nodes] = True

    # 初始化收集器
    all_src = []
    all_dst = []

    for node in least_connected:
        node = int(node)
        current_feature = normalized_features[node]

        # 排除自身、已有邻居和高入度节点
        exclusion_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        exclusion_mask[list(adjacency_dict[node]) + [node]] = True
        exclusion_mask |= high_degree_mask

        candidate_nodes = torch.where(~exclusion_mask)[0]
        if len(candidate_nodes) == 0:
            continue

        # 采样
        sample_size = min(3 * k, len(candidate_nodes))
        sampled_indices = torch.randperm(len(candidate_nodes))[:sample_size]
        sampled_candidates = candidate_nodes[sampled_indices]

        # 计算相似度
        candidate_features = normalized_features[sampled_candidates]
        similarities = torch.mv(candidate_features, current_feature)
        avg_sim = torch.mean(similarities)
        # print('avg_sim:', avg_sim)

        # 动态调整k（sigmoid改进）
        dynamic_scale = torch.sigmoid((avg_sim - 0.1) * 10).item()
        dynamic_scale = min(max(dynamic_scale, 0.1), 1.0)
        current_k = max(1, int(round(dynamic_scale * k)))
        current_k = min(current_k, len(sampled_candidates))
        # print(f'[DEBUG] sim:{avg_sim:.3f} → scale:{dynamic_scale:.2f} → k:{current_k}')

        # 加上相似度过滤
        high_sim_mask = similarities >= avg_sim
        filtered_indices = torch.where(high_sim_mask)[0]

        if len(filtered_indices) >= current_k:
            selected_candidates = sampled_candidates[filtered_indices]
            selected_similarities = similarities[filtered_indices]
            _, top_filtered_indices = torch.topk(selected_similarities, current_k)
            selected_nodes = selected_candidates[top_filtered_indices]
        else:
            _, top_k_indices = torch.topk(similarities, current_k)
            selected_nodes = sampled_candidates[top_k_indices]

        # 检查边存在性
        src_nodes = torch.full((len(selected_nodes),), node, dtype=torch.long)
        exist_mask = graph.has_edges_between(src_nodes, selected_nodes)
        new_edges = selected_nodes[~exist_mask]

        # 收集
        if len(new_edges) > 0:
            all_src.append(src_nodes[~exist_mask])
            all_dst.append(new_edges)
    if all_src:
        all_src = torch.cat(all_src)
        all_dst = torch.cat(all_dst)
        graph.add_edges(all_src, all_dst)

        added_src = all_src
        added_dst = all_dst

        # # 可视化其中一个新增边目的节点的自我网络
        # visualize_ego_network(graph, int(added_dst[0]), "added_node_ego.pdf")

    graph = dgl.add_self_loop(graph)

    return graph, added_src, added_dst


def add_edges_to_nodes(graph):
    # ========== 预计算阶段 ==========
    # 入度分析与k值计算
    in_degrees = graph.in_degrees()
    avg_degree = torch.mean(in_degrees.float()).item()
    k = int(avg_degree * 5)
    print(f'平均参考度数k：{k}')

    # 节点选择
    sorted_nodes = torch.argsort(in_degrees)
    least_connected = sorted_nodes[:int(len(sorted_nodes) * 0.5)]
    print(f'低连接节点数：{len(least_connected)}')

    # 特征归一化 (GPU加速)
    node_features = graph.ndata['feature'].float()
    normalized_features = F.normalize(node_features, p=2, dim=1)

    # 全局相似度矩阵预计算
    with torch.no_grad():
        global_sim = torch.mm(normalized_features, normalized_features.t())

    # 构建排除矩阵 (一次性计算)
    threshold = get_high_degree_threshold(in_degrees, n_std=2)
    high_degree_nodes = (in_degrees > threshold)

    # ========== 批量处理阶段 ==========
    # 准备批量数据
    num_nodes = graph.num_nodes()
    edges_to_add = []

    # 预计算所有排除mask (向量化操作)
    batch_exclude_mask = torch.zeros((len(least_connected), num_nodes), dtype=torch.bool)
    for i, node in enumerate(least_connected):
        predecessors = graph.predecessors(node).long()
        batch_exclude_mask[i, predecessors] = True
        batch_exclude_mask[i, node] = True
        batch_exclude_mask[i, high_degree_nodes] = True

    # 批量相似度处理
    batch_sim = global_sim[least_connected]  # [num_least, num_nodes]
    batch_sim[batch_exclude_mask] = -float('inf')

    # 动态候选选择 (GPU加速)
    sample_size = min(3 * k, num_nodes)
    top_values, top_indices = torch.topk(batch_sim, sample_size, dim=1)

    # ========== 动态调整与边添加 ==========
    for i, node in enumerate(least_connected):
        node = int(node)
        candidates = top_indices[i]
        similarities = top_values[i]

        # 过滤无效候选
        valid_mask = candidates != -1
        candidates = candidates[valid_mask]
        similarities = similarities[valid_mask]

        if len(candidates) == 0:
            continue

        # 动态调整逻辑（向量化计算）
        avg_sim = torch.mean(similarities).item()
        dynamic_scale = min(max((avg_sim + 0.15) / 0.5, 0.1), 1.0)
        current_k = max(1, min(int(dynamic_scale * k), len(candidates)))

        # 质量筛选（高于平均相似度）
        above_avg = similarities > avg_sim
        valid_candidates = candidates[above_avg]

        # 保底机制
        if len(valid_candidates) == 0:
            selected = candidates[:1]
        else:
            selected = valid_candidates[:current_k]

        # 去重检查
        exist_mask = graph.has_edges_between(
            torch.full((len(selected),), node),
            selected
        )
        new_edges = selected[~exist_mask]

        if len(new_edges) > 0:
            edges_to_add.append((node, new_edges))

    # ========== 批量添加边 ==========
    if edges_to_add:
        src = torch.cat([torch.full((len(d),), s) for s, d in edges_to_add])
        dst = torch.cat([d for _, d in edges_to_add])
        graph.add_edges(src, dst)

    return graph


def plot_degree_distribution(graph):
    # 获取节点的度数
    degrees = graph.in_degrees().numpy()

    # 将节点按度数从小到大排序
    sorted_degrees = sorted(degrees)

    # 将节点分为五组
    num_groups = 5
    group_size = len(sorted_degrees) // num_groups  # 每组的节点数量

    # 确保每组的大小正确，如果节点数无法整除5组，将多余的节点分配到前几组
    groups = []
    for i in range(num_groups):
        start = i * group_size
        if i == num_groups - 1:  # 最后一组包含所有剩余的节点
            end = len(sorted_degrees)
        else:
            end = (i + 1) * group_size
        groups.append(sorted_degrees[start:end])

    # 计算每组的度数范围 (最小值和最大值)
    group_ranges = [(group[0], group[-1]) for group in groups]
    group_counts = [len(group) for group in groups]

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    x = [f'{r[0]} - {r[1]}' for r in group_ranges]  # 每个组的度数范围
    plt.bar(x, group_counts)

    # 添加标签和标题
    plt.xlabel('Degree Range')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Nodes in Different Degree Groups')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图形
    plt.show()

import collections
def compute_degree_dis(graph, dataname):
    # 获取所有节点的度（假设是无向图，用 in_degrees 或 out_degrees 均可）
    degrees = graph.in_degrees().tolist()  # 转换为列表

    # 统计度分布
    degree_counts = collections.Counter(degrees)
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in degrees_sorted]
    # 计算概率
    total_nodes = graph.number_of_nodes()
    probabilities = [c / total_nodes for c in counts]

    # 保存为Origin兼容的格式（制表符分隔，含标题行）
    with open(f"{dataname}_degree_distribution.txt", "w") as f:
        f.write("Degree_k\tCount_N_k\tProbability_P_k\n")  # 列标题
        for k, n_k, p_k in zip(degrees, counts, probabilities):
            f.write(f"{k}\t{n_k}\t{p_k:.6f}\n")  # 制表符分隔，概率保留6位小数