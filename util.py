import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def compute_cosine_similarity_and_mi(feature, y, state, n_bins=10):
    """
    计算特征矩阵和标签的余弦相似性与互信息矩阵。
    Args:
        feature (torch.Tensor): 特征矩阵 (n_samples, n_features)
        y (torch.Tensor): 一维标签向量 (n_samples,)
        n_bins (int): 离散化区间数，用于互信息计算
    Returns:
        torch.Tensor, torch.Tensor: 余弦相似性矩阵和互信息矩阵
    """
    # 确保输入是 PyTorch 张量
    if isinstance(feature, np.ndarray):
        feature = torch.tensor(feature, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.int64)

    # Step 1: 将 y 转换为 one-hot 矩阵
    if y.dim() == 1:
        num_classes = torch.max(y).item() + 1  # 获取类别数量
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()  # 转换为 one-hot 矩阵 (n_samples, num_classes)

    # Step 2: 计算余弦相似性
    feature_norm = feature / torch.norm(feature, dim=0, keepdim=True)  # 归一化特征
    y_norm = y_one_hot / torch.norm(y_one_hot, dim=0, keepdim=True)  # 归一化 one-hot 标签
    cosine_similarity_matrix = torch.mm(feature_norm.T, y_norm)       # (n_features, num_classes)
    # np.savetxt('./data/featurecorrelation/feaslab_cos.csv', cosine_similarity_matrix)
    plot_similarity_heatmap(cosine_similarity_matrix, state+"feaslab_cos.pdf", state+' Feature Lable Cos')
    print("feature-label cosine_similarity")
    print(cosine_similarity_matrix)

    cosine_features = torch.mm(feature_norm.T, feature_norm)
    # np.savetxt('./data/featurecorrelation/feas_cos.csv', cosine_features)
    plot_similarity_heatmap(cosine_features, state+"feas_cos.pdf", state+' Features Cos')
    print("feature-feature cosine_similarity")
    print(cosine_features)

    # Step 3: 计算互信息
    # 转为 NumPy 数组进行互信息计算
    feature_np = feature.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()  # y 的原始标签向量
    mi_matrix = []

    for i in range(feature_np.shape[1]):  # 遍历每个特征列
        # 离散化特征
        feature_discretized = np.digitize(feature_np[:, i], bins=np.linspace(np.min(feature_np[:, i]), np.max(feature_np[:, i]), n_bins))
        # 计算互信息 (使用 sklearn 的 mutual_info_classif)
        mi = mutual_info_classif(feature_discretized.reshape(-1, 1), y_np, discrete_features=True)
        mi_matrix.append(mi[0])  # 提取互信息值

    mi_matrix = torch.tensor(mi_matrix).unsqueeze(1)  # 转为 PyTorch 张量并调整形状 (n_features, 1)
    plot_similarity_heatmap(mi_matrix, state+"fealab_mul.pdf", state+' FeatLab Mul')
    print("feature with label")
    print(mi_matrix)

    # mult_information(feature_np, y_one_hot.detach().cpu().numpy(), state)
    # compute_feature_pairwise_mi(feature_np, state)
    person_similarity = binary_pearson_correlation(feature_np, y_np, state)

    return cosine_similarity_matrix, cosine_features


def mult_information(feature, y_one_hot, state):

    mi_matrix = []
    for i in range(y_one_hot.shape[1]):  # 针对每个类别标签
        mi_matrix.append(mutual_info_classif(feature, y_one_hot[:, i], discrete_features=False))
    mi_matrix = torch.tensor(mi_matrix).T
    plot_similarity_heatmap(mi_matrix, state+"feaseachlab.pdf", state+' Feature Label MulInfo')
    print("feature with each label")
    print(mi_matrix)
    return


def compute_feature_pairwise_mi(feature, state):
    """
    计算特征矩阵中所有特征两两之间的互信息量。

    参数:
    - feature: torch.Tensor，形状为 (n_samples, n_features)，特征矩阵。

    返回:
    - mi_matrix: torch.Tensor，形状为 (n_features, n_features)，两两特征的互信息量矩阵。
    """
    # 确保输入为 NumPy 数组
    feature_np = feature.numpy() if isinstance(feature, torch.Tensor) else feature
    n_features = feature_np.shape[1]

    # 初始化互信息矩阵
    mi_matrix = np.zeros((n_features, n_features))

    # 遍历所有特征对 (i, j)
    for i in range(n_features):
        for j in range(i, n_features):  # 仅计算上三角部分，减少重复计算
            if i == j:
                mi_matrix[i, j] = mutual_info_regression(feature_np[:, [i]], feature_np[:, j])[0]
            else:
                mi_value = mutual_info_regression(feature_np[:, [i]], feature_np[:, j])[0]
                mi_matrix[i, j] = mi_value
                mi_matrix[j, i] = mi_value  # 对称赋值

    # 转为 PyTorch 张量
    mi_matrix_torch = torch.tensor(mi_matrix)
    plot_similarity_heatmap(mi_matrix_torch, state+"feas_mult.pdf", state+' Features mulInfo')

    return mi_matrix_torch

def binary_pearson_correlation(feature, y, state):
    """
    计算每个特征与每个类别的皮尔逊相关性系数

    参数:
        feature (numpy.ndarray): 特征矩阵 (n_samples, n_features)
        y (numpy.ndarray): 标签向量 (n_samples, )

    返回:
        pearson_corr_matrix (numpy.ndarray): 每个特征与每个类别的皮尔逊相关性系数 (n_features, n_classes)
    """
    # 检查输入
    if feature.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")
    y = y.flatten()

    # 确定类别数量
    classes = np.unique(y)  # 类别标签
    n_classes = len(classes)  # 类别数
    n_features = feature.shape[1]  # 特征数

    # 初始化存储相关性系数的矩阵
    pearson_corr_matrix = np.zeros((n_features, n_classes))

    # 计算每个类别的皮尔逊相关性系数
    for i, cls in enumerate(classes):
        # 创建类别的二值指示向量
        y_binary = (y == cls).astype(float)

        # 计算中心化的特征和二值标签
        feature_mean = np.mean(feature, axis=0)  # 每个特征的均值
        y_mean = np.mean(y_binary)              # 当前类别指示向量的均值

        feature_centered = feature - feature_mean
        y_centered = y_binary - y_mean

        # 计算皮尔逊相关性分子和分母
        numerator = np.sum(feature_centered * y_centered[:, np.newaxis], axis=0)
        denominator = np.sqrt(np.sum(feature_centered**2, axis=0)) * np.sqrt(np.sum(y_centered**2))

        # 避免除零
        denominator[denominator == 0] = 1e-10

        # 计算当前类别的相关性系数
        pearson_corr_matrix[:, i] = numerator / denominator

    plot_similarity_heatmap(pearson_corr_matrix, state+"fealab_per.pdf", state+"fealab_per")

    return pearson_corr_matrix


def plot_similarity_heatmap(similarity_matrix, filename, name):
    """
    绘制特征相似性矩阵的热力图，并保存为 PDF 文件。

    参数:
    - similarity_matrix: torch.Tensor or np.ndarray，相似性矩阵 (n_features, n_features)。
    - filename: str，保存的 PDF 文件名。
    """
    # 确保相似性矩阵为 NumPy 数组
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.detach().cpu().numpy()

    path = './data/feat-correlation/amazon/{}'.format(filename)

    # 获取矩阵大小，生成坐标标签
    n_features = similarity_matrix.shape[0]
    feature_labels = [f"F{i}" for i in range(n_features)]

    n_xticklabels = similarity_matrix.shape[1]
    y_labels = [f"{i}" for i in range(n_xticklabels)]
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap="coolwarm", fmt=".2f",
                xticklabels=y_labels, yticklabels=feature_labels, cbar_kws={'label': 'Similarity'})
    plt.title(name + " Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    # Adjust label rotation
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)

    # 保存为 PDF
    plt.savefig(path, format="pdf")
    plt.show()
    plt.close()


def select_fea(labelcos, feacos):
    # 参数定义
    labelcos = torch.tensor(labelcos)
    label_thresholds = [0.7, 0.4]  # 标签-特征相似性的阈值
    label_thresholds = torch.tensor(label_thresholds, dtype=torch.float32)
    similarity_threshold_feature = 0.7  # 特征-特征相似性的阈值

    # Step 1: 按阈值筛选特征
    selected_features_mask = labelcos > label_thresholds  # (num_features, num_labels)
    selected_features_per_label = [
        torch.nonzero(selected_features_mask[:, label_idx], as_tuple=True)[0]
        for label_idx in range(label_thresholds.shape[0])
    ]

    # # Step 2: 去除冗余特征
    # final_features_set = set()
    #
    # for label_idx, features in enumerate(selected_features_per_label):
    #     if features.numel() == 0:  # 当前标签没有选中特征
    #         continue
    #
    #     # 获取当前标签的特征-标签相似性得分
    #     label_scores = labelcos[features, label_idx]
    #
    #     # 提取选中特征的相似性子矩阵
    #     feature_similarity = feacos[features][:, features]
    #
    #     # 构建一个布尔向量来标记是否保留某特征
    #     keep_mask = torch.ones(len(features), dtype=torch.bool)
    #
    #     for i in range(len(features)):
    #         if not keep_mask[i]:  # 当前特征已被标记为冗余
    #             continue
    #
    #         for j in range(i + 1, len(features)):
    #             if not keep_mask[j]:  # 冗余特征已被处理
    #                 continue
    #
    #             # 检查特征 i 和特征 j 是否冗余
    #             if feature_similarity[i, j] > similarity_threshold_feature:
    #                 # 比较 labelcos 得分，保留得分高的特征
    #                 if label_scores[i] >= label_scores[j]:
    #                     keep_mask[j] = False  # 删除特征 j
    #                 else:
    #                     keep_mask[i] = False  # 删除特征 i
    #                     break  # 当前特征 i 已删除，跳过检查其他特征
    #
    #     # 根据保留的索引筛选非冗余特征
    #     non_redundant_features = features[keep_mask]
    #
    #     # 将非冗余特征加入全局集合
    #     final_features_set.update(non_redundant_features.tolist())
    #
    # # Step 3: 返回去重后的特征下标列表
    # final_features = sorted(final_features_set)
    final_list = torch.cat(selected_features_per_label).tolist()

    return final_list

def select_features(labelcos):
    # 计算绝对差值
    labelcos = torch.tensor(labelcos)
    abs_diff = torch.abs(labelcos[:, 0] - labelcos[:, 1])  # 比较第 1 列和第 2 列

    # 定义阈值
    threshold = 0.1

    # 筛选超过阈值的特征索引
    selected_features = (abs_diff > threshold).nonzero(as_tuple=True)[0]

    return selected_features