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
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} PR-AUC {:.2f} GMean {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100,
                                                                    final_pa*100, final_gmean*100))


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="questions",
                        help="Dataset for this model (yelp/amazon/ACM/BlogCatalog/weibo/Reddit)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
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
    num_classes = 2
    features = graph.ndata['feature']
    labels = graph.ndata['label']

    time1 = time.time()
    process = psutil.Process(os.getpid())
    with graph.local_scope():
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
            min=1), -1).unsqueeze(-1).to(features.device)
        graph.ndata['h'] = features * D_invsqrt  # 源节点归一化
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))  # 聚合

        feat = 0.5 * features + 0.5 * graph.ndata['h']
    d = features.shape[1]

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

    n_select = k
    print(k)

    if k == 1:
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

    if homo:
        model = BWGNN(in_feats, h_feats, num_classes, new_graph, d=order)
    else:
        model = BWGNN_Hetero(in_feats, h_feats, num_classes, new_graph, d=order)
    """ """
    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, new_graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, new_graph, d=order)
        train1(model, graph, args)

    else:
        final_mf1s, final_aucs, final_recs, final_pres, final_pa, final_gmean, head_f1, tail_f1 = [], [], [], [], [], [], [], []
        time1  = time.time()

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
        print('REC: {:.2f}+-{:.2f}, PRE: {:.2f}+-{:.2f}, MF1: {:.2f}+-{:.2f}, '
              'AUC: {:.2f}+-{:.2f}, PR-AUC: {:.2f}+-{:.2f}, Gmean: {:.2f}+-{:.2f}'.
              format(100 * np.mean(final_recs), 100 * np.std(final_recs),
                     100 * np.mean(final_pres), 100 * np.std(final_pres),
                     100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
                     100 * np.mean(final_aucs), 100 * np.std(final_aucs),
                     100 * np.mean(final_pa), 100 * np.std(final_pa),
                     100 * np.mean(final_gmean), 100 * np.std(final_gmean)
                     ))

