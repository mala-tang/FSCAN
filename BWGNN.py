import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv


class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        self.dropout1 = nn.Dropout(0.3)
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * (I-D^-1/2 A D^-1/2) """
            graph.ndata['h'] = feat * D_invsqrt #Feat * D^-1/2
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h')) # (Feat * D^-1/2) * A
            return feat - graph.ndata.pop('h') * D_invsqrt #Feat - (Feat * D^-1/2) * A * D^-1/2

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            # h = self._theta[0]*feat
            h = feat + self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat #残差连接
                # h = self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
            # h = self.dropout1(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat

        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')#使用 sympy.symbols('x') 创建一个符号变量 x，以便进行符号运算
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))#构造贝塔分布的 PDF 形式的一类变形函数
        coeff = f.all_coeffs()#提取多项式所有的系数，并以列表形式返回
        inv_coeff = []
        for i in range(d+1):#将系数逆序排列并转换为浮点数
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
            else:
                self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*(len(self.conv)), h_feats) #1-求和 len(self.conv)-拼接
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d
        self.dropout = nn.Dropout(0.5)  # 新增Dropout层
        self.dropout1 = nn.Dropout(0.5)  # 新增BatchNorm层
        self.dropout2 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(h_feats)
        self.bn2 = nn.BatchNorm1d(h_feats)
        self.bn3 = nn.BatchNorm1d(h_feats)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout1(h)  # 在激活前应用BatchNorm
        #
        h = self.linear2(h)
        # h = self.bn2(h)  # 在激活前应用BatchNorm
        h = self.act(h)
        # h = self.dropout2(h)

        # h = in_feat

        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
        # for idx, conv in enumerate(self.conv): #求和
            h0 = conv(self.g, h)
            # h0 = self.bn1(h0)
            h_final = torch.cat([h_final, h0], -1)
            # if idx == 0: #求和，而不是拼接
            #     h_final = h0
            # else:
            #     h_final = (h_final + h0)
        # h_final = h_final / len(self.thetas)
        h = self.linear3(h_final)
        h = self.bn3(h)
        emb = self.act(h)
        h = self.dropout(emb)
        h = self.linear4(h)
        return emb, h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat),0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h


# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()
        # print(self.thetas)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        for relation in self.g.canonical_etypes:
            # print(relation)
            h_final = torch.zeros([len(in_feat), 0])
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
                # print(h_final.shape)
            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
