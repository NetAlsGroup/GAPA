import os
import random
from time import time
import networkx as nx
from gapa.utils.DataLoader import Loader
from absolute_path import dataset_path


def init(dataset, detection_rate, seed=None, selected_genes_rate=0.4, pattern="no_sort"):
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(int(random.random() * 10000))
    # Init data_loader
    data_loader = Loader(
        dataset, device
    )
    # Load data
    data_loader.G = nx.read_adjlist(os.path.join(dataset_path, dataset + '.txt'), nodetype=int)
    if pattern == "sort":
        data_loader.A = torch.tensor(nx.to_numpy_array(data_loader.G, nodelist=sorted(list(data_loader.G.nodes()))), device=device)
        data_loader.G = nx.from_numpy_array(data_loader.A.cpu().numpy())
    else:
        data_loader.A = torch.tensor(nx.to_numpy_array(data_loader.G, nodelist=list(data_loader.G.nodes())), device=device)
    data_loader.nodes_num = len(data_loader.A)
    data_loader.nodes = torch.tensor(list(data_loader.G.nodes), device=device)
    data_loader.selected_genes_num = int(selected_genes_rate * data_loader.nodes_num)
    data_loader.k = int(detection_rate * data_loader.nodes_num)
    return data_loader


def SixDST_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, detection_rate=0.1, selected_genes_rate=0.4)
    # mode = "m"
    evaluator = SixDSTEvaluator(
        pop_size=pop_size,
        adj=data_loader.A,
        device=device
    )
    controller = SixDSTController(
        path=f'../experiment_data/CND/SixDST/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and write
        pattern="write",
        # cutoff_tag with "no_cutoff_", "matrixGreedy_cutoff_", "popGreedy_cutoff_"
        # default="popGreedy_cutoff_"
        cutoff_tag="popGreedy_cutoff_",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.6,
        mutate_rate=0.2,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    SixDST(
        mode=mode,
        max_generation=5000,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


def CutOff_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, detection_rate=0.1, pattern="sort")
    # mode = "sm"
    evaluator = CutoffEvaluator(
        pop_size=pop_size,
        graph=data_loader.G,
        nodes=data_loader.nodes,
        device=device
    )
    controller = CutoffController(
        path=f'../experiment_data/CND/Cutoff/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and write
        pattern="write",
        # cutoff_tag with "no_cutoff_", "random_cutoff_", "greedy_cutoff_", "popGreedy_cutoff_", "popGA_cutoff_"
        # default="popGA_cutoff_"
        cutoff_tag="popGreedy_cutoff_",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.5,
        mutate_rate=0.3,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    Cutoff(
        mode=mode,
        max_generation=5000,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size
    )
    end = time()
    print(f"\033[91mCurrent mode is {mode}, Current pop size: {pop_size}\033[0m")
    print(f"\033[91mTotal cost: {end - start}\033[0m")
    torch.cuda.empty_cache()


def TDE_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, detection_rate=0.1, pattern="sort")
    # mode = "m"
    evaluator = TDEEvaluator(
        pop_size=pop_size,
        graph=data_loader.G,
        budget=data_loader.k,
        device=device
    )
    controller = TDEController(
        path=f'../experiment_data/CND/TDE/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and write
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.5,
        mutate_rate=0.3,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    TDE(
        mode=mode,
        max_generation=5000,
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
    # device, _ = mutil_init_device(world_size=2)
    import torch
    from gapa.algorithm.CND.SixDST import SixDSTEvaluator, SixDSTController, SixDST
    from gapa.algorithm.CND.Cutoff import CutoffEvaluator, CutoffController, Cutoff
    from gapa.algorithm.CND.TDE import TDEEvaluator, TDEController, TDE
    from gapa.utils.functions import set_seed, Parsers
    args = Parsers()
    # world_size = args.world_size
    # print(f"Modified world size: {world_size}")
    # "ForestFire_n500"
    if args.method == "SixDST":
        print(f"\033[91m***CND->SixDST***\033[0m")
        SixDST_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "TDE":
        print(f"\033[91m***CND->TDE***\033[0m")
        TDE_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "CutOff":
        print(f"\033[91m***CND->CutOff***\033[0m")
        CutOff_main(args.mode, args.pop_size, args.dataset)
    else:
        raise ValueError(f"No such method {args.method}. Please enter SixDST, TDE or CutOff")

    torch.cuda.empty_cache()

