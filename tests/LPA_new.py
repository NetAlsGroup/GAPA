import os
import random
from time import time
import networkx as nx
from gapa.utils.DataLoader import Loader
from absolute_path import dataset_path


def init(dataset, attack_rate, seed=None, sort=False):
    if seed is None:
        set_seed(int(random.random() * 10000))
    else:
        set_seed(seed)
    # Init data_loader
    data_loader = Loader(
        dataset, device
    )
    # Load data
    if dataset == "dolphins" or dataset == "email-Eu-core":
        data_loader.G = nx.read_gml(os.path.join(dataset_path, dataset, dataset + '.gml'), label="id")
    else:
        data_loader.G = nx.read_adjlist(os.path.join(dataset_path, dataset + '.txt'), nodetype=int)
    if not sort:
        data_loader.A = torch.tensor(nx.to_numpy_array(data_loader.G, nodelist=list(data_loader.G.nodes())), device=device)
    else:
        data_loader.A = torch.tensor(nx.to_numpy_array(data_loader.G, nodelist=sorted(list(data_loader.G.nodes()))), device=device)
        data_loader.G = nx.from_numpy_array(data_loader.A.cpu().numpy())
    data_loader.nodes_num = len(data_loader.A)
    data_loader.nodes = torch.tensor(list(data_loader.G.nodes), device=device)
    data_loader.edges = torch.tensor(list(data_loader.G.edges), device=device)
    data_loader.edges_num = len(data_loader.edges)
    data_loader.selected_genes_num = int(attack_rate * 4 * data_loader.nodes_num)
    data_loader.k = attack_rate
    return data_loader


def EDA_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, attack_rate=0.1, sort=True)
    # mode = "mnm"
    evaluator = EDAEvaluator(
        pop_size=pop_size,
        graph=data_loader.G,
        ratio=0,
        device=device
    )
    controller = EDAController(
        path=f'../experiment_data/LPA/EDA/PyTorch_{mode.upper()}/',
        pattern="write",
        data_loader=data_loader,
        loops=1,
        mutate_rate=0.1,
        pop_size=pop_size,
        num_eda_pop=pop_size,
        device=device
    )
    start = time()
    EDA(
        mode=mode,
        max_generation=500,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


def LPA_GA_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, attack_rate=0.1, sort=True)
    # mode = "s"
    evaluator = GAEvaluator(
        pop_size=pop_size,
        graph=data_loader.G,
        ratio=0,
        device=device
    )
    controller = GAController(
        path=f'../experiment_data/LPA/GA/PyTorch_{mode.upper()}/',
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.7,
        mutate_rate=0.1,
        pop_size=pop_size,
        device=device
    )
    start = time()
    LPA_GA(
        mode=mode,
        max_generation=500,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from gapa.utils.init_device import init_device
    device, world_size = init_device(world_size=2)

    import torch
    from gapa.algorithm.LPA.EDA import EDAEvaluator, EDAController, EDA
    from gapa.algorithm.LPA.LPA_GA import GAEvaluator, GAController, LPA_GA
    from gapa.utils.functions import set_seed, Parsers
    args = Parsers()

    if args.method == "EDA":
        print(f"\033[91m***LPA->EDA***\033[0m")
        EDA_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "GA":
        print(f"\033[91m***LPA->GA***\033[0m")
        LPA_GA_main(args.mode, args.pop_size, args.dataset)
    else:
        raise ValueError(f"No such method {args.method}. Please enter EDA or GA")

    torch.cuda.empty_cache()

