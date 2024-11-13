import torch
import networkx as nx
import os
import random
from test.absolute_path import dataset_path
import pandas as pd
import numpy as np
import torch.multiprocessing as tmp


def SIR_model(graph, alpha, beta, days):
    # color_dict = {"S": "orange", "I": "red", "R": "green"}
    SIR_list = []
    for t in range(0, days):
        updateNetworkState(graph, alpha, beta)  # 对网络状态进行模拟更新
        SIR_list.append(list(countSIR(graph)))  # 计算更新后三种节点的数量

    df = pd.DataFrame(SIR_list, columns=["S", "I", "R"])
    _I = df["I"].values
    # df.plot(figsize=(9, 6), color=[color_dict.get(x) for x in df.columns])
    # plt.show()

    return torch.tensor(np.sum(_I), dtype=torch.float)


def updateNodeState(graph, node, alpha, beta):
    if graph.nodes[node]["state"] == "I":  # 感染者
        p = random.random()  # 生成一个0到1的随机数
        if p < beta:  # gamma的概率恢复
            graph.nodes[node]["state"] = "R"  # 将节点状态设置成“R”
    elif graph.nodes[node]["state"] == "S":  # 易感者
        p = random.random()  # 生成一个0到1的随机数
        k = 0  # 计算邻居中的感染者数量
        for neighbor in graph.adj[node]:  # 查看所有邻居状态，遍历邻居用 G.adj[node]
            if graph.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者，则k加1
                k = k + 1
        if p < 1 - (1 - alpha) ** k:  # 易感者被感染
            graph.nodes[node]["state"] = "I"


def updateNetworkState(graph, alpha, beta):
    for node in graph:  # 遍历图中节点，每一个节点状态进行更新
        updateNodeState(graph, node, alpha, beta)


# 计算三类人群的数量
def countSIR(graph):
    S = 0
    I = 0
    for node in graph:
        if graph.nodes[node]["state"] == "S":
            S = S + 1
        elif graph.nodes[node]["state"] == "I":
            I = I + 1
    return S, I, len(graph.nodes) - S - I


def readData(dataset, rate):
    graph: nx.Graph = nx.read_adjlist(os.path.join(dataset_path, dataset + '.txt'), nodetype=int)
    for _node in graph.nodes():
        graph.nodes[_node]["state"] = "S"
    # origin_I = np.random.choice(len(graph.nodes()), 1 if len(graph.nodes()) / 100 < 1 else int(len(graph.nodes()) / 100), replace=False)
    # origin_I = np.arange(1, 34)
    # for node in origin_I:
    #     graph.nodes[node]["state"] = "I"
    # draw_graph(graph)
    degree = [i[1] for i in graph.degree()]
    avg = np.average(degree)
    _beta = avg / (avg ** 2 - avg)
    _alpha = _beta * rate
    return graph, _alpha, _beta


def iterators(node, graph: nx.Graph, _alpha):
    _G = graph.copy()
    _G.nodes[node]["state"] = "I"
    _I = SIR_model(_G, _alpha, 0.8, 50)
    return _I


class SIR:
    def __init__(self, graph: nx.Graph, device):
        self.graph = graph
        self.device = device
        self.nodes = torch.tensor(list(graph.nodes), device=device)
        self.I_list = torch.tensor([])

    def sumI(self, _alpha, normalize=True):
        # Init multiprocessing
        num_cores = int(tmp.cpu_count())
        print(f"{num_cores} cores in this env...")
        if num_cores <= 4:
            pool = tmp.Pool(num_cores)
        else:
            pool = tmp.Pool(4)

        I_list = []
        r = pool.starmap_async(iterators, [(i, self.graph, _alpha) for i in self.graph.nodes()], callback=I_list.extend)
        r.wait()
        if normalize:
            self.I_list = torch.nn.functional.normalize(input=torch.tensor(I_list, device=self.device), p=2, dim=0)
        else:
            self.I_list = torch.tensor(I_list, device=self.device)

    def calTopI(self, k):
        _top_I = self.I_list.topk(k=k)
        return self.nodes[_top_I.indices], _top_I.indices, _top_I.values


if __name__ == "__main__":
    old_G, alpha, beta = readData("jazz", rate=1.5)
    sir = SIR(graph=old_G)
    sir.sumI(alpha)
    # for node in old_G.nodes():
    #     G = old_G.copy()
    #     G.nodes[node]["state"] = "I"
    #     G, sumI = SIR_model(G, alpha, 0.8, 50)

