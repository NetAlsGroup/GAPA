import os
import random
from time import time
import networkx as nx
from gafama.utils.DataLoader import Loader
from absolute_path import model_path, dataset_path


def init(dataset, model, attack_rate, attack_target="node", retrain=False):
    set_seed(int(random.random() * 10000))
    # Init data_loader
    data_loader = Loader(
        dataset, device
    )
    dataset = load_dataset(dataset, dataset_path, device=device)
    adj, feats, labels, num_edge = dataset.adj, dataset.feats, dataset.labels, len(torch.nonzero(dataset.adj.cpu().to_dense()))
    train_index, val_index, test_index = dataset.train_index, dataset.val_index, dataset.test_index
    num_nodes, num_feats, num_classes = dataset.num_nodes, dataset.num_feats, dataset.num_classes
    ori_feats = feats.bool().float()
    del dataset
    load_set(data_loader.dataset, model, num_nodes=num_nodes, num_edge=num_edge)
    gcn = Classifier(
        model_name=model,
        input_dim=num_feats,
        output_dim=num_classes,
        device=device
    )
    save_path = os.path.join(model_path, f'nc_{model}_{data_loader.dataset}.pt')
    if retrain or not gcn.load_model(save_path):
        gcn.initialize()
        gcn.fit(feats, adj, labels, train_index, val_index, verbose=True)
        output, _ = gcn.predict(feats, adj)
        test_acc, test_correct = gcn.get_acc(output, labels, test_index)
        gcn.save(save_path)

    # 获取网络的精度
    ori_output, ori_sig_output = gcn.predict(feats, adj)
    ori_acc, _ = gcn.get_acc(ori_output, labels, test_index)
    print(f"Model acc: {ori_acc}")
    # Load data
    data_loader.G = nx.Graph(adj.to_dense().cpu().numpy())
    data_loader.adj = adj
    data_loader.test_index = test_index
    data_loader.feats = feats
    data_loader.labels = labels
    data_loader.num_edge = num_edge
    data_loader.train_index = train_index
    data_loader.val_index = val_index
    data_loader.num_nodes = num_nodes
    data_loader.num_feats = num_feats
    data_loader.num_classes = num_classes
    if attack_target == "node":
        data_loader.k = int(attack_rate * num_nodes)
    elif attack_target == "edge":
        data_loader.k = int(attack_rate * num_edge)
    return data_loader, gcn


def NCA_GA_main(mode, pop_size, dataset):
    data_loader, gcn = init(dataset=dataset, model="gcn", attack_rate=0.025, attack_target="edge")
    # mode = "ss"
    evaluator = NCA_GAEvaluator(
        classifier=gcn,
        pop_size=pop_size,
        feats=data_loader.feats,
        adj=data_loader.adj,
        test_index=data_loader.test_index,
        labels=data_loader.labels,
        device=device
    )
    controller = NCA_GAController(
        path=f'../experiment_data/NCA/GA/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and "write"
        pattern="write",
        data_loader=data_loader,
        classifier=gcn,
        loops=1,
        crossover_rate=0.7,
        mutate_rate=0.3,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    NCA_GA(
        mode=mode,
        max_generation=1500,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


def GANI_main(mode, pop_size, dataset):
    data_loader, gcn = init(dataset=dataset, model="gcn", attack_rate=0.05, attack_target="node")
    # mode = "ss"
    surrogate = GCN(nfeat=data_loader.num_feats, nclass=data_loader.num_classes, nhid=16, dropout=0.5, with_relu=False, with_bias=True, device=device).to(device)
    surrogate.fit(data_loader.feats, data_loader.adj, data_loader.labels, data_loader.train_index, data_loader.val_index, patience=30)
    print(f"Surrogate acc: {surrogate.test(idx_test=data_loader.test_index)}")
    controller = SGAController(
        path=f'../experiment_data/NCA/SGA/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and "write"
        pattern="write",
        data_loader=data_loader,
        classifier=gcn,
        loops=1,
        crossover_rate=0.7,
        mutate_rate=0.3,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    SGA(
        mode=mode,
        max_generation=100,
        data_loader=data_loader,
        surrogate=surrogate,
        classifier=gcn,
        controller=controller,
        homophily_ratio=0.7,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from gafama.utils.init_device import init_device
    device, world_size = init_device(world_size=2)

    import torch
    from gafama.algorithm.NCA.NCA_GA import NCA_GA, NCA_GAController, NCA_GAEvaluator
    from gafama.algorithm.NCA.SGA import SGA, SGAController
    from gafama.utils.dataset import load_dataset
    from gafama.DeepLearning.Classifier import Classifier, load_set
    from gafama.utils.functions import set_seed, Parsers
    from deeprobust.graph.defense import GCN
    args = Parsers()

    if args.method == "GANI":
        print(f"\033[91m***NCA->GANI***\033[0m")
        GANI_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "GA":
        print(f"\033[91m***NCA->GA***\033[0m")
        NCA_GA_main(args.mode, args.pop_size, args.dataset)
    else:
        raise ValueError(f"No such method {args.method}. Please enter GANI or GA")

    torch.cuda.empty_cache()

