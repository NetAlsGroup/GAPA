import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll = nn.Linear(in_features, out_features)

        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, adj, dropout=0):

        x = self.ll(x)
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x) if adj.sparse_dim() == 2 else torch.bmm(adj, x)
        else:
            x = torch.matmul(adj, x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)
        return x


class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_layers: list, activation: nn.functional = F.elu):
        super(GCN, self).__init__()

        layers = [in_feat]
        layers.extend(hidden_layers)
        layers.append(out_feat)
        num_layers = len(layers) - 1
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers:
                self.layers.append(
                    GraphConvolution(layers[i], layers[i + 1], activation=activation, dropout=True))
            else:
                self.layers.append(GraphConvolution(layers[i], layers[i + 1]))

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        # x=F.softmax(x, dim=-1)
        return x


class CNN(nn.Module):
    def __init__(self, _L):
        super(CNN, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(
                in_features=32 * _L * _L // 16,
                out_features=1
            )
        )

        self.loss = nn.MSELoss()

    def forward(self, feats):
        x = self.convLayer(feats)
        x = self.classifier(x)
        return x
