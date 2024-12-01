import os
from typing import Optional, Union, Tuple
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.sparse as tsp
from torch.nn import Module
from torch.optim import Optimizer
import torch.optim as optim
from GCN import GCN
from SGC import SGC
from ChebNet import ChebNet
from config import cfg
from gapa.utils.functions import gcn_filter, set_seed

_MODEL = {
    'gcn': GCN,
    'sgc': SGC,
    'chebnet': ChebNet
}
_ACTIVATION = {
    'relu': F.relu,
    'elu': F.elu,
    'leak_relu': F.leaky_relu,
    'tanh': torch.tanh,
    '': None
}
_OPTIMIZER = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}
_SET = {
    'cora': {
        'gcn': {
            'net_set': {
                'hidden_layers': [32],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1000,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [32, 4, 32],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
    },
    'citeseer': {
        'gcn': {
            'net_set': {
                'hidden_layers': [16],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 1000,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            },
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [32, 4, 32],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
    },
    'pubmed': {
        'gcn': {
            'net_set': {
                'hidden_layers': [16],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 1000,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            },
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [32, 4, 32],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
    },
    'dblp': {
        'gcn': {
            'net_set': {
                'hidden_layers': [16],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.4,
                'seed': 42,
                'patience': 150,
                'opt': 'adam',
                'lr': 2e-2,
                'weight_decay': 6e-4,
            },
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [32, 4, 32],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
    },
    'squirrel': {
        'gcn': {
            'net_set': {
                'hidden_layers': [64, 64, 64, 64],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 5000,
                'dropout': 0.1,
                'seed': 42,
                'patience': 200,
                'opt': 'adam',
                'lr': 2e-2,
                'weight_decay': 5e-4,
            },
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [16, 16, 16],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 2e-3,
                'weight_decay': 5e-4,
            }
        },
    },
    'chameleon': {
        'gcn': {
            'net_set': {
                'hidden_layers': [64],
                'activation': 'elu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0,
                'seed': 42,
                'patience': 300,
                'opt': 'adam',
                'lr': 4e-3,
                'weight_decay': 5e-4,
            },
        },
        'sgc': {
            'net_set': {
                'hidden_layers': [16, 2, 2],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 2024,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'chebnet': {
            'net_set': {
                'hidden_layers': [32, 4, 32, 4],
                'activation': 'leak_relu'
            },
            'train_set': {
                'max_epoch': 1500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 200,
                'opt': 'adam',
                'lr': 1e-3,
                'weight_decay': 5e-4,
            }
        },
    },
}


def load_set(dataset: str, model: str, set_net: bool = True, set_train: bool = True, **kwargs):
    """

    :return: None
    """
    if model in _SET[dataset]:
        net_set = _SET[dataset][model]['net_set']
        train_set = _SET[dataset][model]['train_set']
    else:
        raise ValueError(f"No such Model...")

    if set_net:
        cfg.net[model] = net_set
    if set_train:
        cfg.train.max_epoch = train_set['max_epoch']
        cfg.train.patience = train_set['patience']
        cfg.train.dropout = train_set['dropout']
        cfg.train.seed = train_set['seed']

        cfg.train.optimal.name = train_set['opt']
        cfg.train.optimal.lr = train_set['lr']
        cfg.train.optimal.weight_decay = train_set['weight_decay']
        # set_seed(cfg.train.seed)


def load_model(model: str, input_dim: int, output_dim: int,
               device: Optional[torch.device] = None, **kwargs) -> Module:

    assert model in _MODEL, f"Not found model {model}"
    assert kwargs['activation'] in _ACTIVATION, f"Not found activation {kwargs['activation']}"
    kwargs['activation'] = _ACTIVATION[kwargs['activation']]

    return _MODEL[model](input_dim, output_dim, **kwargs).to(device)


def load_optim(opt: str, model: Module, lr: float, weight_decay: float) -> Optimizer:
    """

    """
    assert opt in _OPTIMIZER, f"Not found optimizer {opt}"
    return _OPTIMIZER[opt](model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def reset_parameters(m):
    """

    :return: None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class Classifier:
    def __init__(self, model_name: str, input_dim: int, output_dim: int,
                 device: Optional[torch.device] = None):

        self.model = load_model(
            model=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            **cfg.net[model_name]
        )

        self.optimizer = load_optim(
            opt=cfg.train.optimal.name,
            model=self.model,
            lr=cfg.train.optimal.lr,
            weight_decay=cfg.train.optimal.weight_decay
        )
        self.model_name = model_name
        self.dropout = cfg.train.dropout
        self.device = device
        self.output = None
        self.best_model = deepcopy(self.model.state_dict())


        self.train_iters = cfg.train.max_epoch
        self.patience = cfg.train.patience

    def fit(self, features: Union[Tensor, tsp.Tensor], adj: tsp.Tensor, labels: Tensor, idx_train: Tensor,
            idx_val: Optional[Tensor] = None, verbose=False):

        adj_norm = gcn_filter(adj)
        early_stopping = self.patience
        best_loss_val = 100
        with tqdm(total=self.train_iters) as pbar:
            pbar.set_description(f"Train classifier model -> {self.model_name}: ")
            for i in range(self.train_iters):
                self.model.train()
                self.optimizer.zero_grad()
                output = F.log_softmax(self.model(features, adj_norm, dropout=self.dropout), dim=1)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()
                self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    output = F.log_softmax(self.model(features, adj_norm), dim=1)
                    loss_val = F.nll_loss(output[idx_val], labels[idx_val])

                if verbose and i % 50 == 0:
                    output = torch.argmax(output, dim=1)
                    correct_train = torch.eq(output[idx_train], labels[idx_train]).sum()
                    train_acc = correct_train.item() * 1.0 / idx_train.shape[0]
                    correct_val = torch.eq(output[idx_val], labels[idx_val]).sum()
                    val_acc = correct_val.item() * 1.0 / idx_val.shape[0]
                    # print('Epoch {}, training loss: {} val loss: {} train acc:{} val acc: {}'.format(i, loss_train.item(),
                    #                                                                                  loss_val.item(),
                    #                                                                                  train_acc, val_acc))
                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    self.best_model = deepcopy(self.model.state_dict())
                    self.patience = early_stopping
                else:
                    self.patience -= 1
                if i > early_stopping and self.patience <= 0:
                    break

                pbar.set_postfix(Loss=best_loss_val.item())
                pbar.update(1)

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.model.load_state_dict(self.best_model)

    def save(self, save_path: str):
        torch.save(
            {'state_dict': {k: v.to('cpu') for k, v in self.best_model.items()}},
            save_path
        )

    @torch.no_grad()
    def predict(self, features: Union[Tensor, tsp.Tensor], adj: tsp.Tensor) -> tuple:
        adj_norm = gcn_filter(adj)
        self.model.eval()
        output = self.model(features, adj_norm)
        return torch.argmax(output, dim=1), torch.sigmoid(output)

    @torch.no_grad()
    def get_acc(self, output: Tensor, labels: torch.LongTensor, index: torch.LongTensor) -> Tuple[float, int]:
        correct = torch.eq(output[index], labels[index]).sum()
        acc = correct.item() * 1.0 / index.shape[0]
        return acc, correct.item()

    def initialize(self):
        self.model.apply(reset_parameters)

    def load_model(self, load_path: str) -> bool:
        if os.path.exists(load_path):
            model_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(model_dict['state_dict'])
            return True
        else:
            return False

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)