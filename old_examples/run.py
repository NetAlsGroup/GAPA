# It is important to start with 'if __name__ == "__main__":'
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gapa.utils.init_device import init_device
    device, world_size = init_device(world_size=2)

    import torch
    import networkx as nx
    from tests.Custom import ExampleEvaluator, ExampleController
    from tests.absolute_path import dataset_path
    from gapa.utils.DataLoader import Loader
    from gapa.framework.controller import Start
    from gapa.framework.body import Body

    dataset = "ForestFire_n500"
    data_loader = Loader(
        dataset, device
    )
    # Lode your data
    G = nx.read_adjlist(os.path.join(dataset_path, dataset + '.txt'), nodetype=int)
    data_loader.A = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))), device=device)
    data_loader.G = nx.from_numpy_array(data_loader.A.cpu().numpy())
    data_loader.nodes_num = len(G.nodes())
    budget = int(0.1 * data_loader.nodes_num)
    pop_size = 41
    fit_side = "min"
    body = Body(critical_num=data_loader.nodes_num, budget=budget, pop_size=pop_size, fit_side=fit_side, device=device)
    evaluator = ExampleEvaluator(pop_size, data_loader.A, device)
    controller = ExampleController(budget=budget, pop_size=pop_size, pc=0.5, pm=0.3, side=fit_side, mode="m", num_to_eval=10, device=device)
    controller.dataset = dataset
    Start(max_generation=200, data_loader=data_loader, controller=controller, evaluator=evaluator, body=body, world_size=world_size)
