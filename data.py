from dgl.data import FraudDataset
from dgl.data.utils import load_graphs
import dgl
import warnings
import torch
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from dgl.nn.pytorch.conv import EdgeWeightNorm
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
import dgl.function as fn

warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, name='tfinance', homo=True):
        self.name = name
        graph = None
        prefix = './data'
        if name == 'tfinance':
            graph, label_dict = load_graphs(f'{prefix}/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        elif name == 'tsocial':
            graph, label_dict = load_graphs(f'{prefix}/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudDataset(name, train_size=0.4, val_size=0.2)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        
        elif name == 'amazon':
            dataset = FraudDataset(name, train_size=0.4, val_size=0.2)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)

        elif name == 'ACM' or name == 'ogbn_arixv' or name == 'elliptic' or name == 'BlogCatalog' or name == 'pubmed' or name =='questions':
            data = sio.loadmat("./data/{}.mat".format(name))
            # 提取标签
            label = data['Label'] if 'Label' in data else data['gnd']
            label = label.squeeze()  # 确保是 1D 数组
            # 提取节点特征
            attr = data['Attributes'] if 'Attributes' in data else data['X']
            # 如果 attr 是稀疏矩阵，需要转换为密集矩阵
            if sp.issparse(attr):  # 检查是否为稀疏矩阵
                attr = attr.toarray()  # 转换为密集矩阵（numpy array）
            attr = torch.tensor(attr, dtype=torch.float32)
            network = data['Network'] if ('Network' in data) else data['A']
            if isinstance(network, sp.spmatrix):  # 如果是稀疏矩阵
                network = network.tocoo()
                src = torch.tensor(network.row, dtype=torch.int64)
                dst = torch.tensor(network.col, dtype=torch.int64)
            else:  # 如果是密集矩阵
                network = torch.tensor(network, dtype=torch.float32)
                src, dst = torch.where(network > 0)  # 获取边的索引

            # 构建 DGL 图
            graph = dgl.graph((src, dst))

            # 添加节点特征
            graph.ndata['feature'] = attr
            # 添加节点标签
            graph.ndata['label'] = torch.tensor(label, dtype=torch.int64)
            graph = dgl.add_self_loop(graph)

        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph

