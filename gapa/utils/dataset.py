import os
import random
import sys
import pickle as pkl
from typing import Tuple, Union, Optional
import scipy.sparse as sp
import torch
import numpy as np
import pandas as pd
from gapa.utils.functions import gcn_filter, eye
from tests.absolute_path import dataset_path


_ALL_DATASET = {
    'bin': ['citeseer', 'pubmed'],
    'pt': ['cora', 'dblp'],
    'npz': ['chameleon_filtered', 'squirrel_filtered'],
    'csv': []
}


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _preprocess_features(features: sp.lil_matrix):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = torch.tensor(features.sum(1))
    r_inv = torch.float_power(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv).float()
    features = r_mat_inv.matmul(torch.tensor(features.todense()).float())
    return features.numpy()


class BaseDataset:
    def __init__(self, name: str, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.model = model
        self.device = device
        self.root = os.path.join(dataset_path, name)
        print(self.root)
        assert os.path.exists(self.root), f"Not found {name} dataset"
        if name in _ALL_DATASET['bin']:
            self._bin_init(name)
        elif name in _ALL_DATASET['npz']:
            self._npz_init(name)
        elif name in _ALL_DATASET['pt']:
            self._pt_init(name)
        elif name in _ALL_DATASET['csv']:
            self._csv_init(name)
        else:
            self._other_init(name)

    def _process_filter(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None]:
        """
        计算重归一化拉普拉斯矩阵等提取网络拓扑信息的矩阵

        :return: filter or (filter, filter)
        """
        if self.model is None:
            return None
        else:
            return gcn_filter(self.adj)

    def _bin_init(self, name):
        if name == 'cora':
            name = 'cora_v2'
        objects = []
        # 读取数据集文件
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(self.root, name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(self.root, name))
        test_idx_range = np.sort(test_idx_reorder)

        if name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        # 加载特征
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        self.feats = torch.tensor(
            data=_preprocess_features(features),
            dtype=torch.float,
            device=self.device
        )
        # self.feats = self.feats.to_sparse()

        del features

        # 加载标签
        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        self.labels = torch.tensor(data=np.argmax(onehot_labels, 1), dtype=torch.long, device=self.device)

        self.train_index = torch.tensor(data=[i for i in range(len(y))], dtype=torch.long, device=self.device)
        self.val_index = torch.tensor(data=[i for i in range(len(y), len(y) + 500)], dtype=torch.long,
                                      device=self.device)
        self.test_index = torch.tensor(data=test_idx_range.tolist(), dtype=torch.long, device=self.device)

        del onehot_labels
        # 计算节点数量 标签类别数 特征维度
        self.num_nodes = self.feats.shape[0]
        self.num_classes = torch.max(self.labels).item() + 1
        self.num_feats = self.feats.shape[1]

        # 加载邻接矩阵
        # _indicate 是一个二维列表 第一维表示连边的横坐标 第二维表示连边的纵坐标
        # 例如 (0,0)(1,1)表示为[[0,1][0,1]]
        _indices = torch.tensor(
            list(map(list, zip(*[(_x, _y) for _x, _y_list in graph.items() for _y in _y_list if _x != _y]))),
            dtype=torch.long,
            device='cpu'
        )
        _value = [True] * len(_indices[0])
        # 获取原始有向图和转换后的无向图
        self.ori_adj = torch.sparse_coo_tensor(
            indices=_indices, values=_value, dtype=torch.float, device=self.device
        )
        self.adj = (self.ori_adj + self.ori_adj.t()).coalesce().bool().float()
        self.filter = self._process_filter()
        del _indices, _value

    def _npz_init(self, name):
        loader = np.load(os.path.join(self.root, f"{name}.npz"))
        # loader = np.load(os.path.join(dataset_path, "cornell", f"cornell.npz"))

        feats = loader['node_features']
        edges = loader['edges']
        labels = loader['node_labels']

        train_masks = loader['train_masks'][0]
        train_index = np.where(np.array(train_masks) == 1)[0]
        # print(loader['train_masks'].sum(axis=1))
        val_masks = loader['val_masks'][0]
        val_index = np.where(np.array(val_masks) == 1)[0]
        # print(loader['val_masks'].sum(axis=1))
        test_masks = loader['test_masks'][0]
        test_index = np.where(np.array(test_masks) == 1)[0]
        # print(loader['test_masks'].sum(axis=1))
        # print(np.unique(labels))

        # indices = np.arange(len(feats))
        # np.random.shuffle(indices)
        #
        # train_size = int(0.8 * len(feats))
        # val_size = int(0.1 * len(feats))
        #
        # train_index = indices[:train_size]
        # val_index = indices[train_size:train_size + val_size]
        # test_index = indices[train_size + val_size:]

        adj = np.zeros((len(feats), len(feats)), dtype=int)
        for edge in edges:
            source, target = edge
            adj[source, target] = 1
            adj[target, source] = 1

        feats = sp.lil_matrix(feats)
        self.feats = torch.tensor(
            data=_preprocess_features(feats),
            dtype=torch.float,
            device=self.device
        )

        self.adj = torch.tensor(adj, device=self.device, dtype=torch.float).to_sparse_coo()
        self.labels = torch.tensor(labels, device=self.device)
        self.train_index = torch.tensor(train_index, device=self.device)
        self.val_index = torch.tensor(val_index, device=self.device)
        self.test_index = torch.tensor(test_index, device=self.device)
        if self.adj.is_sparse:
            self.adj = self.adj.coalesce()
        self.filter = self._process_filter()
        # 计算节点数量 标签类别数 特征维度
        self.num_nodes = self.adj.shape[0]
        self.num_classes = torch.max(self.labels).item() + 1
        self.num_feats = self.feats.shape[1]

    def _pt_init(self, name):
        adj_path = os.path.join(self.root, 'adj.pt')
        feats_path = os.path.join(self.root, 'features.pt')
        labels_path = os.path.join(self.root, 'labels.pt')
        train_index_path = os.path.join(self.root, 'train_index.pt')
        val_index_path = os.path.join(self.root, 'val_index.pt')
        test_index_path = os.path.join(self.root, 'test_index.pt')
        attack_index_path = os.path.join(self.root, 'attack_index.pt')

        self.adj = torch.load(adj_path, map_location=self.device)
        self.feats = torch.load(feats_path, map_location=self.device)
        self.labels = torch.load(labels_path, map_location=self.device)
        self.train_index = torch.load(train_index_path, map_location=self.device)
        self.val_index = torch.load(val_index_path, map_location=self.device)
        self.test_index = torch.load(test_index_path, map_location=self.device)
        self.attack_index = torch.load(attack_index_path, map_location=self.device)

        self.filter = self._process_filter()
        if self.adj.is_sparse:
            self.adj = self.adj.coalesce()
        # 计算节点数量 标签类别数 特征维度
        self.num_nodes = self.adj.shape[0]
        self.num_classes = torch.max(self.labels).item() + 1
        self.num_feats = self.feats.shape[1]

    def _csv_init(self, name):
        adj_path = os.path.join(self.root, 'adjacency.csv')
        labels_path = os.path.join(self.root, 'labels.csv')
        # 加载标签
        y = pd.read_csv(labels_path, header=None)
        self.labels = torch.tensor(y.values, dtype=torch.long, device=self.device).squeeze(dim=1)
        del y
        # 加载邻接矩阵
        edge_index = pd.read_csv(adj_path, header=None)
        _indices = torch.from_numpy(edge_index.values).t()
        _values = torch.ones(_indices.shape[1])
        self.adj = torch.sparse_coo_tensor(
            indices=_indices,
            values=_values,
            dtype=torch.float,
            size=(self.labels.shape[0], self.labels.shape[0]),
            device=self.device
        )
        del edge_index, _indices, _values
        # 数据集没有特征 故用单位矩阵替代
        self.feats = eye(m=self.adj.shape[0], dtype=torch.float, device=self.device)
        # 划分数据集
        seed = 42
        _all = [i for i in range(self.labels.shape[0])]
        test_split = 0.8
        val_split = 0.1
        train_split = 1 - test_split - val_split
        random.seed(seed)
        random.shuffle(_all)
        self.train_index = torch.tensor(
            data=_all[0:int(len(_all) * train_split)],
            dtype=torch.long,
            device=self.device
        )
        self.val_index = torch.tensor(
            data=_all[int(len(_all) * train_split): int(len(_all) * (train_split+val_split))]
        )
        self.test_index = torch.tensor(
            data=_all[int(len(_all) * (train_split+val_split)):],
            dtype=torch.long,
            device=self.device
        )
        # 计算节点数量 标签类别数 特征维度
        self.num_nodes = self.adj.shape[0]
        self.num_classes = torch.max(self.labels).item() + 1
        self.num_feats = self.feats.shape[1]

    def _other_init(self, name):
        pass


class Cora(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'cora'
        super(Cora, self).__init__(name=self.name, model=model, device=device)


class Citeseer(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'citeseer'
        super(Citeseer, self).__init__(name=self.name, model=model, device=device)


class PubMed(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'pubmed'
        super(PubMed, self).__init__(name=self.name, model=model, device=device)


class Dblp(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'dblp'
        super(Dblp, self).__init__(name=self.name, model=model, device=device)


class Chameleon(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'chameleon_filtered'
        # self.name = 'chameleon'
        super(Chameleon, self).__init__(name=self.name, model=model, device=device)


class Squirrel(BaseDataset):
    def __init__(self, model: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = 'squirrel_filtered'
        super(Squirrel, self).__init__(name=self.name, model=model, device=device)


_DATASET = {
    'cora': Cora,
    'citeseer': Citeseer,
    'dblp': Dblp,
    'pubmed': PubMed,
    'chameleon': Chameleon,
    'squirrel': Squirrel
}


def load_dataset(dataset: str, model: Optional[str], device: Optional[torch.device] = None) -> BaseDataset:
    """
    加载一个数据集

    :param dataset: 数据集名称
    :param model: 不同的模型作不同的数据预处理
    :param device: 将数据集加载到对应的GPU设备中
    :return: 返回一个改数据集的类
    """
    assert dataset in _DATASET, f"Not found dataset {dataset}"
    return _DATASET[dataset](model=model, device=device)
