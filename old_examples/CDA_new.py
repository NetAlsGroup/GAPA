import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from time import time
import networkx as nx
from gapa.utils.DataLoader import Loader
from absolute_path import dataset_path


def init(dataset, attack_rate):
    set_seed(int(random.random() * 10000))
    # Init data_loader
    data_loader = Loader(
        dataset, device
    )
    # Load data
    data_loader.G = nx.read_gml(os.path.join(dataset_path, dataset, dataset + '.gml'), label="id")
    data_loader.A = torch.tensor(nx.to_numpy_array(data_loader.G, nodelist=sorted(list(data_loader.G.nodes()))), device=device)
    data_loader.nodes_num = len(data_loader.A)
    data_loader.nodes = torch.tensor(list(data_loader.G.nodes), device=device)
    data_loader.selected_genes_num = int(attack_rate * 4 * data_loader.nodes_num)
    data_loader.k = int(attack_rate * data_loader.nodes_num)
    return data_loader


def EDA_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, attack_rate=0.1)
    # mode = "mnm"
    evaluator = EDAEvaluator(
        pop_size=pop_size,
        graph=data_loader.G.copy(),
        nodes_num=data_loader.nodes_num,
        adj=data_loader.A,
        device=device
    )
    controller = EDAController(
        path=f'../experiment_data/CDA/EDA/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and "write"
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.6,
        mutate_rate=0.2,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    EDA(
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


def CGN_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, attack_rate=0.1)
    # mode = "m"
    evaluator = CGNEvaluator(
        pop_size=pop_size,
        graph=data_loader.G.copy(),
        device=device
    )
    controller = CGNController(
        path=f'../experiment_data/CDA/CGN/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and "write"
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.7,
        mutate_rate=0.01,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    CGN(
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


def QAttack_main(mode, pop_size, dataset):
    data_loader = init(dataset=dataset, attack_rate=0.1)
    # mode = "mnm"
    evaluator = QAttackEvaluator(
        pop_size=pop_size,
        graph=data_loader.G.copy(),
        device=device,
    )
    controller = QAttackController(
        path=f'../experiment_data/CDA/QAttack/PyTorch_{mode.upper()}/',
        # pattern with "overwrite" and "write"
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=0.8,
        mutate_rate=0.1,
        pop_size=pop_size,
        device=device,
    )
    start = time()
    QAttack(
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


if __name__ == "__main__":
    from gapa.utils.init_device import init_device
    device, world_size = init_device(world_size=2)

    import torch
    from gapa.algorithm.CDA.EDA import EDAEvaluator, EDAController, EDA
    from gapa.algorithm.CDA.CGN import CGNEvaluator, CGNController, CGN
    from gapa.algorithm.CDA.QAttack import QAttackEvaluator, QAttackController, QAttack
    from gapa.utils.functions import set_seed, Parsers
    args = Parsers()

    if args.method == "CGN":
        print(f"\033[91m***CDA->CGN***\033[0m")
        CGN_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "QAttack":
        print(f"\033[91m***CDA->QAttack***\033[0m")
        QAttack_main(args.mode, args.pop_size, args.dataset)
    elif args.method == "EDA":
        print(f"\033[91m***CDA->EDA***\033[0m")
        EDA_main(args.mode, args.pop_size, args.dataset)
    else:
        raise ValueError(f"No such method {args.method}. Please enter CGN, QAttack or EDA")

    torch.cuda.empty_cache()
