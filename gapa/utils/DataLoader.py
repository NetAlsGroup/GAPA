class Loader:
    def __init__(self, dataset, device, **kwargs):
        # For NCA
        self.adj = None
        self.test_index = None
        self.feats = None
        self.labels = None
        self.num_edge = None
        self.train_index = None
        self.val_index = None
        self.num_nodes = None
        self.num_feats = None
        self.num_classes = None

        # For CND
        self.G = None
        self.dataset = dataset
        self.device = device
        self.nodes_num = None
        self.k = None
        self.A = None
        self.selected_genes_num = None
        self.nodes = None

        # For CDA
        self.edge_number = None
        self.edge_list = None

        # Others
        self.dict = {}
        try:
            self.__dict__.update(kwargs)
        finally:
            pass

        self._DATASET = ["citeseer", "cora"]
        # CDA dataset setting
        self._CDA_CPU_DATASET = ["karate", "dolphins", "football", "hep-th", "LFR500", "LFR1000", "LFR2000", "LFR3000", "LFR4000"]
        self._CDA_GPU_DATASET = ["LFR5000", "polbooks"]
        self.mode = 'cpu'
        self.world_size = None
        self.support = ["cora", "citeseer"]

    @property
    def CDA_GPU_DATASET(self):
        return self._CDA_GPU_DATASET

    @property
    def CDA_CPU_DATASET(self):
        return self._CDA_CPU_DATASET




