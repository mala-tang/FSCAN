import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
import pandas as pd
from data import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from utils_str import *


def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    # df = compute_similarity_scores(g, labels)
    # # Save the DataFrame as a CSV file
    # df.to_csv('./data/S_true.csv', index=False, header=True)
    index = list(range(len(labels)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        logits = model(features)
        # compute_s(graph, logits)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        if not isinstance(labels, np.ndarray):
            label = labels.detach().cpu().numpy()  # 如果是 PyTorch 张量
        if not isinstance(probs, np.ndarray):
            probs = probs.detach().cpu().numpy()  # 如果是 PyTorch 张量
        f1, thres = get_best_f1(label[val_mask], probs[val_mask])
        preds = numpy.zeros_like(label)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(label[test_mask], preds[test_mask])
        tpre = precision_score(label[test_mask], preds[test_mask])
        tmf1 = f1_score(label[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(label[test_mask], probs[test_mask][:, 1]) #.detach().numpy()

        # if e < 3 or (e + 1) % 20 == 0:
        #     df = compute_similarity_scores(g, preds)
        #     # Save the DataFrame as a CSV file
        #     df.to_csv(f'./data/{tmf1:.3f}_S_pred.csv', index=False, header=True)
        #
        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    return final_tmf1, final_tauc


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

#
# def compute_similarity_scores(graph, Y):
#     if isinstance(Y, np.ndarray):
#         Y = torch.tensor(Y, dtype=torch.float32, device=graph.device)
#
#     deg_inv = torch.pow(graph.in_degrees().float(), -1).unsqueeze(-1)  # D^-1
#     # adj_matrix = graph.adjacency_matrix() # 获取邻接矩阵 (稀疏格式)
#     adj_tensor = graph.adjacency_matrix().to_dense()  # 转为张量
#     D_inv_A = deg_inv.squeeze(-1)[:, None] * adj_tensor  # D^-1 A
#     identity_matrix = torch.eye(adj_tensor.size(0), device=adj_tensor.device)
#     L = identity_matrix - D_inv_A
#     S = L @ Y.float()
#
#     # Convert the tensor to a NumPy array
#     S_np = S.detach().numpy()  # Move to CPU if it's on GPU
#     # Convert Y to a NumPy array (Y could be a matrix with shape (N, K), not a vector)
#     Y_np = Y.detach().numpy()
#     # Ensure both arrays are at least 2D
#     if S_np.ndim == 1:
#         S_np = S_np[:, None]  # Reshape to (n, 1)
#     if Y_np.ndim == 1:
#         Y_np = Y_np[:, None]  # Reshape to (n, 1)
#
#     # Ensure that S_np and Y_np have the same number of rows (N)
#     assert S_np.shape[0] == Y_np.shape[0], "Number of rows in S and Y must be the same"
#
#     # Concatenate S and Y together
#     S_with_Y = np.hstack((S_np, Y_np))
#
#     # Columns for S and Y
#     columns_S = [f'S{i + 1}' for i in range(S_with_Y.shape[1] - Y_np.shape[1])]  # Columns for S
#     columns_Y = [f'Y{i + 1}' for i in range(Y_np.shape[1])]  # Columns for Y # Handle 1D or 2D Y
#
#     # Combine the columns for S, Y, and hetero
#     columns = columns_S + columns_Y# + columns_hetero
#
#     # Create DataFrame with column names
#     df = pd.DataFrame(S_with_Y, columns=columns)
#     print(df)
#     return df
#
# def compute_s(graph, Y):
#     # 计算 D^-1（入度的逆）
#     deg_inv = torch.pow(graph.in_degrees().float(), -1).to(graph.device)  # D^-1
#     deg_inv[deg_inv == float('inf')] = 0  # 防止0度节点导致无穷大
#
#     # 获取稀疏邻接矩阵
#     adj_matrix = graph.adjacency_matrix()  # 邻接矩阵 (稀疏格式)
#     # 将 SparseMatrix 转换为 PyTorch 的稀疏 COOrdinate 张量（如果图库提供了转换方法）
#     adj_matrix_coo = adj_matrix.coalesce()
#
#     # 计算 D^-1 * A (稀疏矩阵乘法)
#     # 使用 shape 获取维度
#     rows = adj_matrix.shape[0]
#     # 构造 D^-1 为稀疏对角矩阵
#     D_inv = torch.sparse_coo_tensor(
#         indices=torch.arange(rows, device=graph.device).repeat(2).view(2, -1),  # 对角线索引
#         values=deg_inv,  # 对角线上的值
#         size=(rows, rows),  # 矩阵大小
#         device=graph.device
#     )
#
#     # 计算 D^-1 * A (稀疏矩阵乘法)
#     D_inv_A = torch.sparse.mm(D_inv, adj_matrix_coo)  # 稀疏矩阵乘法
#
#     # 构造稀疏单位矩阵 I
#     identity_matrix = torch.sparse_coo_tensor(
#         indices=torch.arange(rows, device=graph.device).repeat(2).view(2, -1),  # 对角线索引
#         values=torch.ones(rows, device=graph.device),  # 对角线值为1
#         size=(rows, rows),  # 矩阵大小
#         device=graph.device
#     )
#     # 计算 L = I - D^-1 * A
#     L = identity_matrix - D_inv_A  # 稀疏减法
#
#     # 稀疏矩阵与稠密矩阵的乘法：S = L @ Y
#     S = torch.sparse.mm(L, Y)
#     print(S)
#     return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="Reddit",
                        help="Dataset for this model (yelp/amazon/ACM/BlogCatalog/Facebook/Reddit)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--del_ratio", type=float, default=0.02, help="delete ratios")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    del_ratio = args.del_ratio
    graph = Dataset(dataset_name, homo).graph
    # graph = Dataset(dataset_name, del_ratio, homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    print("原始边数:", graph.num_edges())
    remove_edges_by_label_degree(graph, graph.ndata['label'])
    print("删除异配边边数:", graph.num_edges())
    num_classes = 2

    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
