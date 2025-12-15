import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time

from utils_str import *
from dgl import ops
import sklearn.feature_selection as skfs

from data import Dataset

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix, \
    precision_recall_curve, auc
from BWGNN import *
from sklearn.model_selection import train_test_split

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(model, g, args, train_mask, val_mask, test_mask):
    features = g.ndata['feature']
    labels = g.ndata['label']
    # index = list(range(len(labels)))

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_pa, final_gmean = 0., 0., 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
        precision, recall, _ = precision_recall_curve(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
        # 计算 PR-AUC
        pa = auc(recall, precision)
        gmean_value = Gmean(labels[test_mask], preds[test_mask])

        if best_f1 < f1:
            best_f1 = tmf1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_pa = pa
            final_gmean = gmean_value
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec * 100,
                                                                     final_tpre * 100, final_tmf1 * 100,
                                                                     final_tauc * 100))
    return final_tmf1, final_tauc, final_trec, final_tpre, final_pa, final_gmean


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
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

def mi_agg(graph, features, ori_labels, train_idx):
    #对称归一化会同时降低高度数邻居 j 的贡献，而随机游走归一化仅根据发送节点 i 的度数分配权重。

    graph = dgl.remove_self_loop(graph)
    degrees = graph.out_degrees().float()
    #计算图中每条边的两个端点出度的乘积（即 𝑑𝑢⋅𝑑𝑣） ops.u_mul_v 是 DGL 中用于边级操作的接口，表示对边的源节点 u 和目标节点 v 分别取出 degrees 并做乘积。
    degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
    epsilon = 1e-7
    coefs = 1 / (degree_edge_products ** 0.5 + epsilon)
    #用上述系数加权邻居特征并求和，实现邻居特征的聚合。
    # ops.u_mul_e_sum: 对每条边，从源节点 u 提取 features[u]，乘以边的权重 coefs，然后对每个目标节点 v 聚合（求和）：
    feat_agg = ops.u_mul_e_sum(graph, features, coefs)
    # if train_idx is not None:
    #     feat_agg = feat_agg[train_idx].cpu()
    #     ori_labels = ori_labels[train_idx].cpu()
    feat_agg = feat_agg[train_idx].cpu()
    ori_labels = ori_labels[train_idx].cpu()
    mi_nei_lst = skfs.mutual_info_classif(feat_agg, ori_labels)
    # hom_res = torch.tensor(mi_nei_lst)
    return mi_nei_lst

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="elliptic",
                        help="Dataset for this model (yelp/amazon/ACM/BlogCatalog/ogbn_arixv/elliptic)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--del_ratio", type=float, default=0.02, help="delete ratios")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=10, help="Running times")

    set_random_seed(666)
    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    del_ratio = args.del_ratio
    graph = Dataset(dataset_name, homo).graph
    # graph = Dataset(dataset_name, del_ratio, homo).graph
    num_classes = 2
    features = graph.ndata['feature']
    print("feature shape:", features.shape)
    labels = graph.ndata['label']
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


    #特征选择
    # 获取特征重要性
    time1 = time.time()
    feature_importances = mi_agg(graph, features, labels, idx_train)

    # 方法1：直接按数量选择前N%的特征（更精确）
    sorted_indices = np.argsort(feature_importances)[::-1]  # 降序排列索引
    n_features = features.shape[1]

    percent = 90
    n_select = int(n_features * percent / 100)
    # 选择特征索引
    if percent == 100:
        selected_features = slice(None)  # 选择所有特征
    else:
        selected_features = sorted_indices[:n_select]
        selected_features = np.sort(selected_features).copy()  # 按升序排列索引
    print('selected_features', selected_features)
    features = features[:, selected_features]
    print(features.shape)
    # 更新图中的节点特征
    graph.ndata['feature'] = features
    in_feats = features.shape[1]
    time2 = time.time()
    print('TFI time', time2 - time1)
    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args, train_mask, val_mask, test_mask)

    else:
        final_mf1s, final_aucs, final_recs, final_pres, final_pa, final_gmean = [], [], [], [], [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, AUC, rec, pre, pa, gmean = train(model, graph, args, train_mask, val_mask, test_mask)
            final_mf1s.append(mf1)
            final_aucs.append(AUC)
            final_pres.append(pre)
            final_recs.append(rec)
            final_pa.append(pa)
            final_gmean.append(gmean)
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
                     100 * np.mean(final_gmean), 100 * np.std(final_gmean)))
