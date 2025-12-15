import dgl
import psutil
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from dgl import ops

from find_node import *

from sklearn.ensemble import RandomForestClassifier

from data import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix, \
    precision_recall_curve, auc
from BWGNN import *
from sklearn.model_selection import train_test_split
from utils_str import *
from utils_vis import *
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(model, g, train_mask, val_mask, test_mask, args):
    features = g.ndata['feature']
    labels = g.ndata['label']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_pa, final_gmean = 0., 0., 0., 0., 0., 0., 0., 0.
    best_vpa, best_vgmean = 0., 0.
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        emb, logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits_eval = model(features)
        probs = logits_eval.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        label = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
        preds = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.array(preds)
        vprecision, vrecall, _ = precision_recall_curve(label[val_mask], probs[val_mask][:, 1].detach().numpy())
        # 计算 PR-AUC
        vpa = auc(vrecall, vprecision)
        vgmean_value = Gmean(label[val_mask], preds[val_mask])


        trec = recall_score(label[test_mask], preds[test_mask])
        tpre = precision_score(label[test_mask], preds[test_mask])
        tmf1 = f1_score(label[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(label[test_mask], probs[test_mask][:, 1].detach().numpy())
        precision, recall, _ = precision_recall_curve(label[test_mask], probs[test_mask][:, 1].detach().numpy())
        # 计算 PR-AUC
        pa = auc(recall, precision)
        gmean_value = Gmean(label[test_mask], preds[test_mask])

        if best_f1+ best_vpa + best_vgmean < f1 + vpa + vgmean_value: #best_f1+ best_vpa < f1 + vpa: #best_f1+ best_vpa + best_vgmean < f1 + vpa + vgmean_value:
            best_f1 = f1
            best_vpa = vpa
            best_vgmean = vgmean_value
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_pa = pa
            final_gmean = gmean_value
            tsne_visualize(features, labels, test_mask, title="t-SNE of Original Features "+ args.dataset)
            tsne_visualize(emb, labels, test_mask, title="t-SNE of Learned Embeddings " + args.dataset)
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    # degrees = g.in_degrees()  # 选择入度作为示例
    # test_degree = degrees[test_mask]
    # degree_stratified_analysis(test_degree, label[test_mask], preds[test_mask], 20)
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    return final_tmf1, final_tauc, final_trec, final_tpre, final_pa, final_gmean


def train1(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.5,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)#, weight_decay=1e-4
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_pa, final_gmean = 0., 0., 0., 0., 0., 0., 0., 0.
    best_vpa, best_vgmean = 0., 0.
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        emb, logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        label = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
        preds = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.array(preds)
        vprecision, vrecall, _ = precision_recall_curve(label[val_mask], probs[val_mask][:, 1].detach().numpy())


        trec = recall_score(label[test_mask], preds[test_mask])
        tpre = precision_score(label[test_mask], preds[test_mask])
        tmf1 = f1_score(label[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(label[test_mask], probs[test_mask][:, 1].detach().numpy())
        precision, recall, _ = precision_recall_curve(label[test_mask], probs[test_mask][:, 1].detach().numpy())
        # 计算 PR-AUC
        pa = auc(recall, precision)
        gmean_value = Gmean(label[test_mask], preds[test_mask])

        if best_f1 + best_vpa + best_vgmean < tmf1 + pa + gmean_value:  # best_f1+ best_vpa < f1 + vpa: #best_f1+ best_vpa + best_vgmean < f1 + vpa + vgmean_value: best_f1 < f1:#
            # best_f1 = f1
            # best_vpa = vpa
            # best_vgmean = vgmean_value
            best_f1 = tmf1
            best_vpa = pa
            best_vgmean = gmean_value
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_pa = pa
            final_gmean = gmean_value
            # degrees = g.in_degrees()  # 选择入度作为示例
            # test_degree = degrees[test_mask]
            # head_f1, tail_f1 = head_tail_degree_error_analysis(test_degree, labels[test_mask], preds[test_mask])
            tsne_visualize(features, labels, test_mask, title="t-SNE of Original Features " + args.dataset)
            tsne_visualize(emb, labels, test_mask, title="t-SNE of Learned Embeddings " + args.dataset)
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} PR-AUC {:.2f} GMean {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100,
                                                                    final_pa*100, final_gmean*100))
    # degrees = g.in_degrees()  # 选择入度作为示例
    # test_degree = degrees[test_mask]
    # degree_stratified_analysis(test_degree, label[test_mask], preds[test_mask], 20)

    return final_tmf1, final_tauc, final_trec, final_tpre, final_pa, final_gmean, head_f1, tail_f1

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        preds = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.array(preds)
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def Gmean(y_true, y_pred):
    """binary geometric mean of  True Positive Rate (TPR) and True Negative Rate (TNR)

    Args:
            y_true (np.array): label
            y_pred (np.array): prediction
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    for sample_true, sample_pred in zip(y_true, y_pred):
        TP += sample_true * sample_pred
        TN += (1 - sample_true) * (1 - sample_pred)
        FP += (1 - sample_true) * sample_pred
        FN += sample_true * (1 - sample_pred)

    return math.sqrt(TP * TN / (TP + FN) / (TN + FP))



def degree_stratified_analysis(degrees, labels, preds, step=10):
    """
    基于度数百分比分层的性能分析
    :param degrees: 节点度数数组
    :param labels: 真实标签数组
    :param preds: 预测标签数组
    :param step: 分层步长百分比（10或20）
    """
    # 参数校验
    assert step in [10, 20], "Step参数必须为10或20"
    total_nodes = len(degrees)

    # 生成分层分位数
    percentiles = np.arange(0, 100 + step, step)
    thresholds = np.percentile(degrees, percentiles)

    # 构建分层掩码
    performance_records = []
    for i in range(len(percentiles) - 1):
        # 获取当前分层的上下界
        lower_bound = thresholds[i]
        upper_bound = thresholds[i + 1]

        # 处理最后一个分层的闭合区间
        if i == len(percentiles) - 2:
            layer_mask = (degrees >= lower_bound) & (degrees <= upper_bound)
        else:
            layer_mask = (degrees >= lower_bound) & (degrees < upper_bound)

        # 跳过空分层
        layer_indices = np.where(layer_mask)[0]
        if len(layer_indices) == 0:
            continue

        # 计算分层指标
        layer_degrees = degrees[layer_mask]
        layer_labels = labels[layer_mask]
        layer_preds = preds[layer_mask]

        record = {
            'percentile_range': f"{percentiles[i]}%-{percentiles[i + 1]}%",
            'degree_range': f"{layer_degrees.min():.1f}-{layer_degrees.max():.1f}",
            'node_count': len(layer_indices),
            'node_ratio': len(layer_indices) / total_nodes,
            'f1_score': f1_score(layer_labels, layer_preds, average='macro')
        }
        performance_records.append(record)

    # 打印格式化结果
    print(f"\n{' Degree Stratified Performance Analysis ':=^80}")
    print(f"Step Size: {step}% | Total Nodes: {total_nodes}")
    print("-" * 80)
    print(f"{'Percentile':<12} {'Degree Range':<14} {'Nodes':<8} {'Ratio':<8} {'F1-Score':<10}")
    print("-" * 80)
    for record in performance_records:
        print(f"{record['percentile_range']:<12} {record['degree_range']:<14} "
              f"{record['node_count']:<8} {record['node_ratio']:<8.2%} "
              f"{record['f1_score']:.4f}")
    print("=" * 80)

    return performance_records

def to_numpy(arr):
    return arr.cpu().numpy() if hasattr(arr, 'cpu') else arr

def head_tail_degree_error_analysis(degrees, labels, preds):
    degree = to_numpy(degrees)
    labels = to_numpy(labels)
    preds = to_numpy(preds)

    assert len(degrees) == len(labels) == len(preds), "数组长度不一致"
    total_nodes = len(degrees)
    total_misclassified = np.sum(labels != preds)

    threshold = get_high_degree_threshold(degrees, n_std=1)#Blog-85, questions-95 #amazon-70
    head_mask = degrees > threshold
    tail_mask = degrees <= threshold

    partitions = {
        'Head (High-degree)': head_mask,
        'Tail (Low-degree)': tail_mask
    }

    # 初始化F1值
    head_f1 = None
    tail_f1 = None

    print(f"\nTotal Nodes: {total_nodes} | Total Misclassified: {total_misclassified} "
          f"| Misclassification Rate: {total_misclassified / total_nodes:.2%}")
    print(f"\n{' Head/Tail Node Classification Summary ':=^100}")
    print(f"Degree threshold (95th percentile): {threshold:.2f}")
    print(f"{'Zone':<22} {'F1 Score':<10} {'Correct':<10} {'Wrong':<10} {'Total':<10}")
    print("-" * 70)

    for name, mask in partitions.items():
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        sub_degrees = degree[mask]
        total = len(sub_labels)

        error_mask = sub_labels != sub_preds
        correct = np.sum(~error_mask)
        wrong = np.sum(error_mask)
        f1 = f1_score(sub_labels, sub_preds, average='macro')

        # 记录F1值
        if name == 'Head (High-degree)':
            head_f1 = f1
        elif name == 'Tail (Low-degree)':
            tail_f1 = f1

        print(f"{name:<22} {f1:<10.4f} {correct:<10} {wrong:<10} {total:<10}")

        if wrong == 0:
            print(f"  ➤ No misclassified nodes in {name}.")
            continue

        print(f"  ➤ Misclassified Node Degree Distribution ({'3-bin' if name == 'Head (High-degree)' else '5-bin'} in {name})")
        error_degrees = sub_degrees[error_mask]
        sub_total_degrees = sub_degrees

        zone_min = sub_degrees.min()
        zone_max = sub_degrees.max()
        bin_count = 3 if name == 'Head (High-degree)' else 5
        bin_edges = np.linspace(zone_min, zone_max, bin_count + 1)

        print(f"{'Subrange':<20} {'Err Count':<10} {'Total Nodes':<14} {'Err Rate in Bin':<16}")
        print("-" * 70)

        low_bin_range = None
        low_bin_mask = None
        low_bin_error_mask = None
        for i in range(bin_count):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i < bin_count - 1:
                bin_mask = (sub_degrees > lower) & (sub_degrees <= upper)
            else:
                bin_mask = (sub_degrees > lower) & (sub_degrees <= upper + 1e-6)

            err_bin_mask = bin_mask & error_mask
            total_in_bin = np.sum(bin_mask)
            err_in_bin = np.sum(err_bin_mask)
            err_rate = err_in_bin / total_in_bin if total_in_bin > 0 else 0.0

            print(f"{f'({lower:.1f}, {upper:.1f}]':<20} {err_in_bin:<10} {total_in_bin:<14} {err_rate:<16.2%}")

            if name == 'Tail (Low-degree)' and i == 0:
                low_bin_range = (lower, upper)
                low_bin_mask = bin_mask
                low_bin_error_mask = err_bin_mask

        print()

        if name == 'Tail (Low-degree)' and low_bin_range is not None:
            print(f"  ➤ Fine-grained analysis of the **lowest-degree** bin in Tail nodes (3-bin)")

            low_start, low_end = low_bin_range
            low_bin_degrees_all = sub_degrees[low_bin_mask]
            low_bin_error_degrees = sub_degrees[low_bin_error_mask]

            sub_edges = np.linspace(low_start, low_end, 4)

            print(f"{'Sub-subrange':<20} {'Err Count':<10} {'Total Nodes':<14} {'Err Rate in Bin':<16}")
            print("-" * 70)

            for i in range(3):
                lower = sub_edges[i]
                upper = sub_edges[i + 1]

                if i < 2:
                    bin_sub_mask = (low_bin_degrees_all > lower) & (low_bin_degrees_all <= upper)
                    err_sub_mask = (low_bin_error_degrees > lower) & (low_bin_error_degrees <= upper)
                else:
                    bin_sub_mask = (low_bin_degrees_all > lower) & (low_bin_degrees_all <= upper + 1e-6)
                    err_sub_mask = (low_bin_error_degrees > lower) & (low_bin_error_degrees <= upper + 1e-6)

                total_sub = np.sum(bin_sub_mask)
                err_sub = np.sum(err_sub_mask)
                err_rate = err_sub / total_sub if total_sub > 0 else 0.0

                print(f"{f'({lower:.1f}, {upper:.1f}]':<20} {err_sub:<10} {total_sub:<14} {err_rate:<16.2%}")

            print()

    print("=" * 100)

    # 返回两个区域的F1值
    return head_f1, tail_f1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="elliptic",
                        help="Dataset for this model (yelp/amazon/ACM/BlogCatalog/weibo/Reddit)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    #Amazon 32
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    # parser.add_argument("--imp_ratio", type=float, default=0.3, help=" important ratios of feature")
    #[14, 13,  9, 21, 16, 22,  8, 15,  7,  5, 18, 19, 12, 23, 17]
    # parser.add_argument("--high_deg_ratio", type=float, default=0.95, help="chose the high ratios degree node")
    # parser.add_argument("--del_ratio", type=float, default=1, help="delete ratios")
    # parser.add_argument("--low_deg_ratio", type=float, default=0.01, help="chose the high ratios degree node")
    # parser.add_argument("--add_edges", type=int, default=10, help="the number of add edges")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=10, help="Running times")

    set_random_seed(666)
    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    homo = args.homo
    order = args.order

    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo).graph
    print('graph:', graph)
    # graph = Dataset(dataset_name, del_ratio, homo).graph
    num_classes = 2
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    # imp_ratio = args.imp_ratio
    # important_feature_indices = svd_orthogonal_selection2(features, imp_ratio)

    # with graph.local_scope():
    #     D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
    #         min=1), -0.5).unsqueeze(-1).to(features.device)
    #     graph.ndata['h'] = features * D_invsqrt  # 源节点归一化
    #     graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))  # 聚合
    #     feat = 0.8 * features + 0.2 * graph.ndata['h'] * D_invsqrt
    time1 = time.time()
    process = psutil.Process(os.getpid())
    with graph.local_scope():
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
            min=1), -1).unsqueeze(-1).to(features.device)
        graph.ndata['h'] = features * D_invsqrt  # 源节点归一化
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))  # 聚合

        feat = 0.5 * features + 0.5 * graph.ndata['h']
    d = features.shape[1]
    """
    # # 对特征矩阵进行 SVD
    U, S, Vh = torch.linalg.svd(feat)
    # vital = U[:, :k]
    V = Vh.T
    # top_k_vectors = V[:k, :]  # 选取前 k 行
    # # 计算原始特征的贡献度：对前 k 行取绝对值求和
    feature_importances = torch.abs(V).sum(dim=0)
    feature_importances = np.array(feature_importances)"""
    # 对特征矩阵进行 SVD

    U, S, Vh = torch.linalg.svd(feat, full_matrices=False)
    V = Vh.T  # (f, r)

    # 1. 计算实际秩 r
    tol = 1e-10
    k = (S > tol).sum().item()

    # 3. 取前 k 个奇异值和向量
    sigma2 = S[:k] ** 2        # (k,)
    V_k = V[:, :k]             # (f, k)

    # 4. 计算 α_i
    feature_importances = (V_k**2 * sigma2.unsqueeze(0)).sum(dim=1)

    # 转为 numpy
    feature_importances = feature_importances.cpu().numpy()
    sorted_indices = np.argsort(feature_importances)[::-1]  # 降序排列索引
    n_features = features.shape[1]

    # n_select = int(n_features * imp_ratio)
    n_select = k #重投修改
    print(k)
    # 选择特征索引
    # if imp_ratio == 1:
    if k == 1: #重投修改
        selected_features = slice(None)  # 选择所有特征
    else:
        selected_features = sorted_indices[:n_select]
        selected_features = np.sort(selected_features).copy()  # 按升序排列索引
    features = features[:, selected_features]
    graph.ndata['feature'] = features
    time3 = time.time()
    print('SC time:', time3 - time1)
    print(f'Final selected {len(selected_features)} features: {selected_features}')
    # print(graph.ndata['feature'].shape)
    in_feats = graph.ndata['feature'].shape[1]
    # """  """
    graph_after_deletion, deleted_src, deleted_dst = filter_edge_bisimilarity_optimized(graph, graph.ndata['feature'])
    # print('graph_after_deletion:', graph_after_deletion)
    new_graph, added_src, added_dst = add_edges_to_least_connected(graph_after_deletion)
    # print('new_graph:', new_graph)
    # 结构分析与导出
    # analyze_graph_changes(graph, graph_after_deletion, new_graph,
    #                       (deleted_src, deleted_dst), (added_src, added_dst))

    if homo:
        model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
    else:
        model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
    """ """
    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train1(model, graph, args)

    else:
        final_mf1s, final_aucs, final_recs, final_pres, final_pa, final_gmean, head_f1, tail_f1 = [], [], [], [], [], [], [], []
        time1  = time.time()
        # 初始化

        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, new_graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, new_graph, d=order)
            mf1, AUC, rec, pre, pa, gmean, head, tail = train1(model, new_graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(AUC)
            final_pres.append(pre)
            final_recs.append(rec)
            final_pa.append(pa)
            final_gmean.append(gmean)
            head_f1.append(head)
            tail_f1.append(tail)
        time2 = time.time()
        print('Spend time: {:.4f}s'.format(time2 - time1))
        # CPU内存
        cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
        print(f"CPU: {cpu_mem:.2f} MB")
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        final_pres = np.array(final_pres)
        final_recs = np.array(final_recs)
        final_pa = np.array(final_pa)
        final_gmean = np.array(final_gmean)
        # final_head = np.array(head_f1)
        # final_tail = np.array(tail_f1)
        # print('REC: {:.2f}+-{:.2f}, PRE: {:.2f}+-{:.2f}, MF1: {:.2f}+-{:.2f}, '
        #       'AUC: {:.2f}+-{:.2f}, PR-AUC: {:.2f}+-{:.2f}, Gmean: {:.2f}+-{:.2f},'
        #       'Head_F1: {:.2f}+-{:.2f}, Tail_F1: {:.2f}+-{:.2f}'.
        #       format(100 * np.mean(final_recs), 100 * np.std(final_recs),
        #              100 * np.mean(final_pres), 100 * np.std(final_pres),
        #              100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
        #              100 * np.mean(final_aucs), 100 * np.std(final_aucs),
        #              100 * np.mean(final_pa), 100 * np.std(final_pa),
        #              100 * np.mean(final_gmean), 100 * np.std(final_gmean),
        #              100 * np.mean(final_head), 100 * np.std(final_head),
        #              100 * np.mean(final_tail), 100 * np.std(final_tail)
        #              ))
        print('REC: {:.2f}+-{:.2f}, PRE: {:.2f}+-{:.2f}, MF1: {:.2f}+-{:.2f}, '
              'AUC: {:.2f}+-{:.2f}, PR-AUC: {:.2f}+-{:.2f}, Gmean: {:.2f}+-{:.2f}'.
              format(100 * np.mean(final_recs), 100 * np.std(final_recs),
                     100 * np.mean(final_pres), 100 * np.std(final_pres),
                     100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
                     100 * np.mean(final_aucs), 100 * np.std(final_aucs),
                     100 * np.mean(final_pa), 100 * np.std(final_pa),
                     100 * np.mean(final_gmean), 100 * np.std(final_gmean)
                     ))

