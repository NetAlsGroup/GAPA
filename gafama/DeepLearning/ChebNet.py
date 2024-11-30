import torch
from torch import nn
from torch.nn import functional as F
import dgl
import dgl.nn.pytorch.conv as dglnn


class ChebNet(nn.Module):
    def __init__(self, in_feats, out_feat, hidden_layers: list, activation=F.relu, **kwargs):
        super().__init__()
        self.input_linear = nn.Linear(in_feats, hidden_layers[0])
        self.act = activation
        self.chebconv = dglnn.ChebConv(hidden_layers[0], hidden_layers[0], hidden_layers[1], activation=self.act)
        self.output_linear = nn.Linear(hidden_layers[2], out_feat)

    def forward(self, x, adj, dropout=0):
        src, dst = adj.to_dense().nonzero(as_tuple=True)
        graph = dgl.graph((src, dst))
        x = self.input_linear(x)
        x = self.act(x)
        if dropout:
            x = F.dropout(x, dropout)
        x = self.chebconv(graph, x, lambda_max=[2])
        if dropout:
            x = F.dropout(x, dropout)
        x = self.output_linear(x)
        return x


if __name__ == "__main__":
    # 生成一个 5x5 的随机对称邻接矩阵
    torch.manual_seed(42)
    adj_matrix = torch.randint(0, 2, (5, 5))
    adj_matrix = torch.triu(adj_matrix, 1)
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix.fill_diagonal_(0)

    print("Adjacency Matrix:")
    print(adj_matrix)

    # src, dst = adj_matrix.nonzero(as_tuple=True)
    #

    # g = dgl.graph((src, dst))
    #

    # print(g)


