import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def find_best_deleted_node(graph, original_graph, deleted_edges, labels):
    src, dst = deleted_edges
    deleted_map = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        deleted_map.setdefault(v, []).append(u)

    best_node = -1
    best_score = -1

    for node, neighbors in deleted_map.items():
        original_neighbors = set(original_graph.predecessors(node).tolist())
        degree = len(original_neighbors)
        if degree == 0:
            continue

        diff_label_count = sum(labels[n] != labels[node] for n in neighbors)
        ratio = diff_label_count / degree
        score = ratio / degree  # 引入度数惩罚

        if score > best_score:
            best_score = score
            best_node = node

    return best_node, deleted_map.get(best_node, [])

# 查找加边最优节点
def find_best_added_node(graph, original_graph, added_edges, labels):
    src, dst = added_edges
    added_map = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        added_map.setdefault(u, []).append(v)

    best_node = -1
    best_ratio = -1

    for node, neighbors in added_map.items():
        new_neighbors = set(graph.successors(node).tolist())
        same_label_count = sum(labels[n] == labels[node] for n in neighbors)
        ratio = same_label_count / len(new_neighbors) if new_neighbors else 0
        if ratio > best_ratio:
            best_ratio = ratio
            best_node = node

    return best_node, added_map.get(best_node, [])

# 导出选中节点的结构（删边/加边）为GEXF
def export_selected_node_subgraph(original_graph, modified_graph, node_id, neighbors,
                                  mode='deletion', output_path='subgraph.gexf'):
    G = nx.DiGraph()

    labels = original_graph.ndata['label']
    G.add_node(node_id, label=int(labels[node_id].item()))

    for n in neighbors:
        G.add_node(n, label=int(labels[n].item()))
        if mode == 'deletion':
            # 入边: 从 n -> node_id
            if modified_graph.has_edges_between(n, node_id):
                G.add_edge(n, node_id, weight=1)
            else:
                G.add_edge(n, node_id, weight=0)
        elif mode == 'addition':
            # 出边: 从 node_id -> n
            if original_graph.has_edges_between(node_id, n):
                G.add_edge(node_id, n, weight=0)  # 原始边
            else:
                G.add_edge(node_id, n, weight=1)  # 新添加边

    nx.write_gexf(G, output_path)
    print(f"[INFO] Exported subgraph of node {node_id} to {output_path}")

