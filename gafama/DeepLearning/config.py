from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = 'reddit'
_C.model = ''
_C.role = ''
_C.hidden_layers = 4
_C.activation = ''
_C.pretrain = False
_C.net = CN()

_C.net.gcn = CN()
_C.net.gcn.hidden_layers = 4
_C.net.gcn.activation = ''

_C.train = CN()
_C.train.max_epoch = 1000
_C.train.patience = 30
_C.train.dropout = 0.5
_C.train.seed = 42

_C.train.optimal = CN()
_C.train.optimal.name = 'adam'
_C.train.optimal.lr = 1e-3
_C.train.optimal.weight_decay = 0
"""Attack"""
_C.attack = CN()
_C.attack.name = ''
_C.attack.seed = 42
"""TDGIA"""
_C.attack.tdgia = CN()
_C.attack.tdgia.lr = 1
_C.attack.tdgia.control_factor = 4
_C.attack.tdgia.step = 0.2
_C.attack.tdgia.scaling = 1
_C.attack.tdgia.alpha = 0.33
_C.attack.tdgia.weight1 = 0.9
_C.attack.tdgia.weight2 = 0.1
_C.attack.tdgia.max_add = 500
_C.attack.tdgia.max_connection = 100
_C.attack.tdgia.max_epoch = 2001
"""NETTACK"""
_C.attack.nettack = CN()

cfg = _C
