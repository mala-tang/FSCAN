import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import seaborn as sns
import os

from sklearn.manifold import TSNE


def compute_part_sim(feature, label, idx):
    n = label.shape[0]
    print(f'number of nodes:{n}')
    label_matrix = torch.zeros(n, 2, dtype=torch.float)
    label_matrix.scatter_(1, label.unsqueeze(1).long(), 1)  # 确保标签正确转为one-hot
    label_norm = F.normalize(label_matrix, p=2, dim=0)  # 标签列归一化

    # 特征按列归一化（假设特征矩阵为 (n_samples, n_features)）
    feature_norm = F.normalize(feature, p=2, dim=0)  # 按列归一化每个特征

    # 计算余弦相似度
    cosine_sim = torch.mm(feature_norm.t(), feature_norm)  # (d_features, 2)

    # ----------------------------------------
    # 3. 计算皮尔逊相关系数 (d×2矩阵)
    # ----------------------------------------
    # 方法1：利用中心化后的余弦相似性
    def pearson_corr(x, y):
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        return F.cosine_similarity(x_centered.t().unsqueeze(1),
                                   y_centered.t().unsqueeze(0), dim=2)
    d = feature.shape[1]
    save_dir = './data/feat-correlation/amazon'
    plot_heatmap(
        cosine_sim,
        title='Feature-Feature Cosine Similarity',
        fname=os.path.join(save_dir, 'part_feature_feature_cosine.pdf'),
        xtick_labels=[f'{i}' for i in idx],
        ytick_labels=[f'{i}' for i in idx],
        figsize=(8, 8),
        vmin=None,
        vmax=None
    )

    # pearson_corr_matrix = pearson_corr(feature, label_matrix)
    # plot_part_heatmap(cosine_sim, 'Feature-Feature Cosine Similarity', './data/feat-correlation/amazon/part_feature_sim.pdf', d)
    # plot_heatmap(cosine_sim, 'Feature-Label Cosine Similarity', './data/feat-correlation/amazon/cosine_sim_all.pdf', d)
    # plot_heatmap(pearson_corr_matrix, 'Feature-Label Pearson Correlation', './data/feat-correlation/amazon/pearson_corr_all.pdf', d)
#
#
# def plot_part_heatmap(matrix, title, fname, idx):
#     plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(
#         matrix.numpy(),
#         annot=False,
#         cmap='coolwarm',
#         center=0,
#         vmin=-1,
#         vmax=1,
#         xticklabels=['Normal', 'Abnormal'],
#         yticklabels=[f'{i}' for i in range(int(idx))]
#         # yticklabels=[f'{index}' for index in idx]  # y轴直接用重要特征的索引值
#     )
#     ax.set_title(title)
#     ax.set_xlabel('Label Class')
#     ax.set_ylabel('Feature Index')
#
#     # 设置纵坐标标签方向
#     plt.yticks(
#         rotation=0,  # 水平显示
#         fontsize= 6,  # 根据特征数量调整字号
#         va='center'  # 垂直居中
#     )
#
#     # 设置PDF保存参数
#     plt.savefig(
#         fname,
#         format='pdf',  # 指定格式为PDF
#         bbox_inches='tight',  # 去除白边
#         pad_inches=0.1,  # 保留少量边距
#         dpi=300,  # 保持高分辨率（用于栅格元素）
#         metadata={  # 添加PDF元数据
#             'Title': title,
#             'Author': 'Auto-generated',
#             'Subject': 'Feature-Label Correlation Analysis'
#         }
#     )
#     plt.show()
#     plt.close()

def compute_sim(feature, label, save_dir='./data/feat-correlation/amazon/'):
    n = label.shape[0]
    d = feature.shape[1]
    print(f'Number of nodes: {n}, feature dim: {d}')
    os.makedirs(save_dir, exist_ok=True)

    # 创建one-hot标签矩阵
    label_matrix = torch.zeros(n, 2, dtype=torch.float)
    label_matrix.scatter_(1, label.unsqueeze(1).long(), 1)

    # 特征按列归一化（特征为 n × d）
    feature_norm = F.normalize(feature, p=2, dim=0)  # 列归一化
    label_norm = F.normalize(label_matrix, p=2, dim=0)  # 标签列归一化

    # 1. 特征-特征余弦相似度 (d × d)
    feat_feat_cos_sim = torch.mm(feature_norm.t(), feature_norm)

    # 2. 特征-标签余弦相似度 (d × 2)
    feat_label_cos_sim = torch.mm(feature_norm.t(), label_norm)

    part_feat_label_cos_sim = torch.mm(feature_norm.t(), label_norm)


    # # 3. 特征-标签皮尔逊相关系数 (d × 2)
    # def pearson_corr(x, y):
    #     x_centered = x - x.mean(dim=0, keepdim=True)
    #     y_centered = y - y.mean(dim=0, keepdim=True)
    #     return F.cosine_similarity(x_centered.t().unsqueeze(1),
    #                                y_centered.t().unsqueeze(0), dim=2)

    # feat_label_pearson_corr = pearson_corr(feature, label_matrix)

    # # 绘图
    plot_heatmap(
        feat_feat_cos_sim,
        title='Feature-Feature Cosine Similarity',
        fname=os.path.join(save_dir, 'feature_feature_cosine.pdf'),
        xtick_labels=[f'{i}' for i in range(d)],
        ytick_labels=[f'{i}' for i in range(d)],
        figsize=(8, 8),
        vmin=None,
        vmax=None
    )

    # plot_heatmap(
    #     feat_label_cos_sim,
    #     title='Feature-Label Cosine Similarity',
    #     fname=os.path.join(save_dir, 'feature_label_cosine.pdf'),
    #     xtick_labels=['0', '1'],
    #     ytick_labels=[f'{i}' for i in range(d)],
    #     figsize=(6, 8),
    #     vmin=None,
    #     vmax=None
    # )

    # plot_heatmap(
    #     feat_label_pearson_corr,
    #     title='Feature-Label Pearson Correlation',
    #     fname=os.path.join(save_dir, 'feature_label_pearson.pdf'),
    #     xtick_labels=['Normal', 'Abnormal'],
    #     ytick_labels=[f'{i}' for i in range(d)],
    #     figsize=(4, 6),
    #     vmin=None,
    #     vmax=None
    # )


# def plot_heatmap(matrix, title, fname, xtick_labels, ytick_labels, figsize=(6, 6), vmin=None, vmax=None):
#     plt.figure(figsize=figsize)
#     ax = sns.heatmap(
#         matrix.detach().cpu().numpy(),
#         annot=False,
#         cmap='coolwarm',
#         xticklabels=xtick_labels,
#         yticklabels=ytick_labels,
#         vmin=vmin if vmin is not None else matrix.min().item(),
#         vmax=vmax if vmax is not None else matrix.max().item(),
#         square=True,
#         cbar_kws={"shrink": 0.8}
#     )
#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel('Label/Feature', fontsize=10)
#     ax.set_ylabel('Feature Index', fontsize=10)
#     plt.xticks(rotation=0, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#
#     plt.savefig(
#         fname,
#         format='pdf',
#         bbox_inches='tight',
#         pad_inches=0.1,
#         dpi=300,
#         metadata={
#             'Title': title,
#             'Author': 'Auto-generated',
#             'Subject': 'Feature Correlation Heatmap'
#         }
#     )
#     plt.show()
#     plt.close()

def plot_heatmap(matrix, title, fname, xtick_labels, ytick_labels, figsize=(7, 8), vmin=None, vmax=None):
    # 设置字体
    mpl.rcParams['font.family'] = 'Arial'

    # 动态设置颜色范围
    vmin = vmin if vmin is not None else matrix.min().item()
    vmax = vmax if vmax is not None else matrix.max().item()

    # 创建画布，避免色条与主图过近
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[22, 1], wspace=0.2)

    # 主图
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])  # 色条

    sns.heatmap(
        matrix.detach().cpu().numpy(),
        ax=ax,
        cbar_ax=cbar_ax,
        cmap='coolwarm',
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        vmin=vmin,
        vmax=vmax,
        square=False,
        linewidths=0,
        linecolor='none'
    )

    # 标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold', fontname='Arial', pad=12)
    ax.set_xlabel('Feature Index', fontsize=14, fontweight='bold', fontname='Arial') # Feature Index
    ax.set_ylabel('Feature Index', fontsize=14, fontweight='bold', fontname='Arial')

    # 坐标轴刻度
    ax.set_xticklabels(xtick_labels, fontsize=12, fontname='Arial', rotation=0)
    ax.set_yticklabels(ytick_labels, fontsize=12, fontname='Arial', rotation=0)

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in cbar_ax.spines.values():
        spine.set_visible(False)

    # 去除色条刻度线外框
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.yaxis.label.set_size(14)
    cbar_ax.yaxis.label.set_fontname('Arial')

    plt.savefig(
        fname,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300,
        metadata={
            'Title': title,
            'Author': 'Auto-generated',
            'Subject': 'Feature Correlation Heatmap'
        }
    )
    plt.show()
    plt.close()

import matplotlib.colors as mcolors
def darken_color(color, amount=0.6):
    """
    将颜色变深，amount 越小颜色越深
    """
    c = mcolors.to_rgb(color)
    return (c[0] * amount, c[1] * amount, c[2] * amount)


def tsne_visualize(features, labels, mask, title):
    """
    features: Tensor or ndarray, shape [N, D]
    labels: Tensor or ndarray, shape [N]
    mask: boolean Tensor or ndarray, shape [N]
    save_path: 若不为 None，则保存为 PDF 文件，例如 "tsne_features.pdf"
    """
    save_path = os.path.join('./data/feat-correlation/', f'{title}.pdf')
    # ---- 数据转 numpy ----
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    # ---- 只取 test_mask ----
    feats = features[mask]
    labs = labels[mask]

    # ---- t-SNE ----
    tsne = TSNE(n_components=2, random_state=42)
    feats_2d = tsne.fit_transform(feats)

    # ---- 可视化 ----
    plt.figure(figsize=(7, 6))

    colors = ['C0', 'C1']
    markers = ['o', '^']
    text_labels = ['Normal', 'Abnormal']

    for cls in [0, 1]:
        idx = (labs == cls)
        edge_col = darken_color(colors[cls], amount=0.5)

        plt.scatter(
            feats_2d[idx, 0],
            feats_2d[idx, 1],
            c=colors[cls],
            marker=markers[cls],
            label=text_labels[cls],
            alpha=0.9,
            edgecolor=edge_col,
            linewidths=0.8,
            s=10
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # ---- 保存为 PDF ----
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path = save_path + ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()
