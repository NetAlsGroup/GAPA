import torch.nn as nn
from torch.nn import functional as F
import dgl.nn.pytorch.conv as dglnn
import dgl


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, activation=F.relu, **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))

    def forward(self, h, is_graph=True, dropout=0):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                if dropout:
                    h = F.dropout(h, dropout)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h


class SGC(nn.Module):
    def __init__(self, in_feats, out_feat, hidden_layers: list, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mlp = None
        mlp_layers = hidden_layers[1]
        if mlp_layers == 1:
            self.sgc = dglnn.SGConv(in_feats, out_feat, k=hidden_layers[2])
        else:
            self.sgc = dglnn.SGConv(in_feats, hidden_layers[0], k=hidden_layers[2])
            self.mlp = MLP(hidden_layers[0], out_feat, out_feat, mlp_layers-1, activation=activation)

    def forward(self, x, adj, dropout=0):
        src, dst = adj.to_dense().nonzero(as_tuple=True)
        graph = dgl.graph((src, dst))
        x = self.sgc(graph, x.to_dense())
        if dropout:
            x = F.dropout(x, dropout)
        if self.mlp is not None:
            x = self.mlp(x, False, dropout)
        return x
