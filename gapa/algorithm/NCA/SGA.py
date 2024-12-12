import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import torch.distributed as dist
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import AS_Rate, current_time, init_dist, Acc
from gapa.utils.functions import gcn_filter, tensorToSparse, adjReshapeAddDim


class SGAEvaluator(BasicEvaluator):
    def __init__(self, feats, adj, test_index, labels, pop_size, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.edge_list = None
        self.feats = feats.clone()
        self.adj = adj.to_dense().clone()
        self.test_index = test_index
        self.labels = labels
        self.W = None

    def forward(self, population):
        device = population.device
        W, feats, adj, labels, test_index, edge_list = self._to_device(device)
        fw = feats @ W
        edge_scores = torch.tensor([], device=device)
        for i in population:
            current_edge = edge_list[i].flatten()
            ori_node = current_edge[0]
            n_added_node_list = current_edge[1::2]
            adj[ori_node, n_added_node_list] = 1
            adj[n_added_node_list, ori_node] = 1
            adj_norm = gcn_filter(adj.to_sparse_coo())
            logit = adj_norm @ adj_norm @ fw
            predict_class = logit.argmax(1)
            surrogate_losses = -torch.ne(predict_class[test_index], labels[test_index]).sum()
            edge_scores = torch.cat((edge_scores, surrogate_losses.unsqueeze(0)))
            adj[ori_node, n_added_node_list] = 0
            adj[n_added_node_list, ori_node] = 0
        return edge_scores

    def _to_device(self, device):
        W = self.W.clone().to(device)
        feats = self.feats.clone().to(device)
        adj = self.adj.clone().to(device)
        labels = self.labels.clone().to(device)
        test_index = self.test_index.clone().to(device)
        edge_list = self.edge_list.clone().to(device)
        return W, feats, adj, labels, test_index, edge_list


class SGABody(Body):
    def __init__(self, pop_size, homophily_decrease_score, edge_list, budget, fit_side, device):
        super().__init__(
            critical_num=len(edge_list),
            budget=budget,
            pop_size=pop_size,
            fit_side=fit_side,
            device=device
        )
        self.homophily_decrease_score = homophily_decrease_score.clone()
        self.edge_list = edge_list.clone()
        self.device = device

    def elitism(self, population, mutation_population, fitness_list, new_fitness_list, best_fitness_list):
        stack_population = torch.vstack((population, mutation_population))
        stack_fitness_list = torch.hstack((fitness_list, new_fitness_list))
        top_index = torch.argsort(stack_fitness_list)[:self.pop_size]
        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index].int()
        edges = self.edge_list[population]
        scores = self.homophily_decrease_score[edges[:, :, 1]]
        scores = torch.sum(scores, 1)
        top_index = torch.argsort(scores, descending=True)
        max_score = scores[top_index[0]]
        max_scores_index = (scores == max_score).nonzero(as_tuple=True)[0]
        first_max_score = max_scores_index.min()
        return population, fitness_list, self.edge_list[population[first_max_score]], fitness_list[first_max_score]

    def solve_conflict(self, population):
        tmp_pop = self.edge_list[population].clone()
        current_valid = []
        over_count = 0
        for i in range(len(tmp_pop)):
            edges = tmp_pop[i, :, 1].tolist()
            remove_dup_edges = list(set(edges))

            while len(remove_dup_edges) != len(edges):
                still_need = len(edges) - len(remove_dup_edges)
                candidate_id = torch.arange(len(self.edge_list), device=self.device)
                new_add = self.edge_list[torch.randint(0, len(candidate_id), (still_need,), dtype=torch.long, device=self.device), 1].clone()
                remove_dup_edges = list(set(remove_dup_edges + new_add.tolist()))
                over_count += 1
                if over_count > 100:
                    over_count = 0
                    break

            tmp_pop[i, :len(remove_dup_edges), 1] = torch.tensor(remove_dup_edges[:len(tmp_pop[i])], device=self.device)
            edges = tmp_pop[i, :, 1].tolist()
            remove_dup_edges = sorted(edges)

            while remove_dup_edges in current_valid:
                still_need = len(tmp_pop[i])
                candidate_id = torch.arange(len(self.edge_list), device=self.device)
                new_add = self.edge_list[torch.randint(0, len(candidate_id), (still_need,), dtype=torch.long, device=self.device), 1].clone()
                remove_dup_edges = sorted(list(set(new_add.tolist())))
                over_count += 1
                if over_count > 100:
                    over_count = 0
                    break

            tmp_pop[i, :len(remove_dup_edges), 1] = torch.tensor(remove_dup_edges[:len(tmp_pop[i])], device=self.device)
            edges = tmp_pop[i, :, 1].tolist()
            current_valid.append(sorted(edges))

        all_edges = self.edge_list.clone().unsqueeze(1).unsqueeze(2)  # Size (len(all_edges), 1, 1, 2)
        tmp_pop = tmp_pop.unsqueeze(0)  # Size (1, pop_ize, attack_limit, 2)
        matches = (all_edges == tmp_pop).all(dim=3)
        population = torch.argmax(matches.int(), dim=0)
        return population


class SGAController(BasicController):
    def __init__(self, path, pattern, data_loader, classifier, loops, crossover_rate, mutate_rate, pop_size, device, fit_side="min"):
        super().__init__(
            path,
            pattern,
        )
        self.loops = loops
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.device = device
        self.fit_side = fit_side
        self.dataset = data_loader.dataset
        self.budget = data_loader.k
        self.classifier = classifier
        self.edge_list = None
        self.edge_num = None
        self.mode = None
        self.adj = None
        self.feats = None
        self.test_index = None
        self.labels = None
        self.homophily_decrease_score = None

    def setup(self, data_loader, evaluator: SGAEvaluator, W, edge_list=None):
        copy_graph = data_loader.G.copy()
        if edge_list is None:
            self.edge_num = len(copy_graph.edges())
            self.edge_list = torch.tensor(list(copy_graph.edges()), device=evaluator.device)
        else:
            self.edge_list = edge_list.clone()
            self.edge_num = len(edge_list)
        self.feats = data_loader.feats
        self.adj = evaluator.adj.clone()
        self.test_index = data_loader.test_index
        self.labels = data_loader.labels
        evaluator.edge_list = self.edge_list.clone()
        evaluator.W = W
        return evaluator

    def calculate(self, max_generation, evaluator):
        elite_edge, elite_edge_score = None, None
        body = SGABody(self.pop_size, self.homophily_decrease_score, self.edge_list, self.budget, self.fit_side, evaluator.device)
        for loop in range(self.loops):
            ONE, population = body.init_population()
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(population)
            best_fitness_list = torch.tensor(data=[], device=self.device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.min(fitness_list)))
            for generation in range(max_generation):
                new_population1 = population.clone()
                new_population2 = body.selection(population, fitness_list)
                crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                crossover_population = body.solve_conflict(crossover_population)
                mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                mutation_population = body.solve_conflict(mutation_population)
                new_fitness_list = evaluator(mutation_population)
                population, fitness_list, elite_edge, elite_edge_score = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
        return elite_edge, elite_edge_score

    def mp_calculate(self, rank, max_generation, evaluator, world_size, return_dict):
        device = init_dist(rank, world_size)
        crossover_population = None
        elite_edge, elite_edge_score = None, None
        edge_list = self.edge_list.to(device)
        homophily_decrease_score = self.homophily_decrease_score.to(device)

        component_size = self.pop_size // world_size
        body = SGABody(component_size, homophily_decrease_score, edge_list, self.budget, self.fit_side, device)
        for loop in range(self.loops):
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            best_fitness_list = torch.tensor(data=[], device=device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(component_fitness_list)))

            if rank == 0:
                population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)]
                fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for _ in range(world_size)]
            else:
                population = None
                fitness_list = None

            dist.gather(component_population, population, dst=0)
            dist.gather(component_fitness_list, fitness_list, dst=0)
            if rank == 0:
                population = torch.cat(population)
                fitness_list = torch.cat(fitness_list)

            for generation in range(max_generation):
                if rank == 0:
                    new_population1 = population.clone()
                    new_population2 = body.selection(new_population1, fitness_list)
                    body.pop_size = self.pop_size
                    crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=new_population1.dtype, device=device)
                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                    crossover_population = body.solve_conflict(crossover_population)
                    body.pop_size = component_size
                if rank == 0:
                    crossover_population = torch.stack(torch.split(crossover_population, component_size))
                else:
                    crossover_population = torch.stack( [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)])
                dist.broadcast(crossover_population, src=0)
                mutation_population = body.mutation(crossover_population[rank], self.mutate_rate, ONE)
                mutation_population = body.solve_conflict(mutation_population)
                new_component_fitness_list = evaluator(mutation_population).to(device)
                if rank == 0:
                    elitism_population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)]
                    elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for _ in range(world_size)]
                else:
                    elitism_population = None
                    elitism_fitness_list = None
                dist.gather(mutation_population, elitism_population, dst=0)
                dist.gather(new_component_fitness_list, elitism_fitness_list, dst=0)
                if rank == 0:
                    elitism_population = torch.cat(elitism_population)
                    elitism_fitness_list = torch.cat(elitism_fitness_list)
                    body.pop_size = self.pop_size
                    population, fitness_list, elite_edge, elite_edge_score = body.elitism(population, elitism_population, fitness_list, elitism_fitness_list, best_fitness_list)
                    top_index = torch.argsort(fitness_list)[:component_size]
                    component_population = population[top_index]
                    component_fitness_list = fitness_list[top_index]
                    body.pop_size = component_size
                else:
                    component_population = torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device)
                    component_fitness_list = torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device)
                dist.broadcast(component_population, src=0)
                dist.broadcast(component_fitness_list, src=0)
        if rank == 0:
            return_dict[rank] = [elite_edge.cpu(), elite_edge_score.cpu()]
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()


class SGAAlgorithm:
    def __init__(self, data_loader, surrogate, classifier, idx_test, homophily_ratio, zeroone_features, device):
        self.device = device
        self.dataset = data_loader.dataset
        self.features = data_loader.feats.clone()
        self.modified_features = data_loader.feats.clone().to_sparse_coo()
        self.adj = data_loader.adj.clone()
        self.modified_adj = data_loader.adj.clone()
        self.labels = data_loader.labels
        self.features_dim = data_loader.num_feats
        self.idx_test = idx_test
        self.features_id = None
        self.feature_avg = None
        self.n_class = data_loader.num_classes
        self.classes = list(set(self.labels.clone().cpu()))
        self.nnodes = data_loader.adj.shape[0]
        self.surrogate = surrogate
        self.classifier = classifier
        self.zeroone_features = zeroone_features
        self.mean_degree = int(data_loader.adj.sum() / self.nnodes)
        self.average_features_nums = np.diff(tensorToSparse(data_loader.feats.to_sparse_coo()).indptr).mean()
        self.get_sorted_features(zeroone_features)
        self.major_features_nums = int(self.average_features_nums)
        self.major_features_candidates = self.features_id[:, :self.major_features_nums]
        sample_array = torch.zeros(self.nnodes, device=device)
        current_degree = self.adj.to_dense().sum(1).int()
        for i in range(self.nnodes):
            # maximal link-attack budget: max(current degree, 2 * mean degree)
            mean_d = (self.mean_degree + 1) * 2
            if current_degree[i] >= mean_d:
                sample_array[i] = mean_d
            elif current_degree[i] == 0:
                continue
            else:
                sample_array[i] = current_degree[i]
        self.sample_array1 = sample_array
        self.ratio = homophily_ratio

        # calculate the original homophily information of each node
        tmp: torch.Tensor = data_loader.labels.clone()
        score = []
        decrease_score = []
        for i in range(len(tmp)):
            current_neighbors = torch.where(data_loader.adj.to_dense()[i] == 1)[0]
            current_neighbors_length = len(current_neighbors)
            if current_neighbors_length == 0:
                current_neighbors_length = 1
            current_neighbors_same_label_length = len(torch.where(tmp[current_neighbors] == tmp[i])[0])
            score.append(current_neighbors_same_label_length / current_neighbors_length)
            decrease_score.append(current_neighbors_same_label_length / (current_neighbors_length * current_neighbors_length + current_neighbors_length))
        decrease_score = torch.tensor(decrease_score, device=device)
        index = torch.tensor(np.argsort(-decrease_score.cpu().numpy()), device=device)
        self.homophily_index = index
        self.homophily_score = score
        self.homophily_decrease_score = decrease_score

        self.injected_nodes_classes = torch.tensor([], device=device)
        self.injected_nodes_origin = torch.tensor([], device=device)
        self.ori_nodes = torch.tensor([], device=device)
        self.W = None
        self.n_added_labels = None

    def main(self, data_loader, controller, max_generation, world_size):
        best_acc = []
        best_asr = []
        time_list = []
        self.W = self.get_linearized_weight()
        n_added = data_loader.k
        selected_degree_distribution = self.sample_array1[torch.randperm(len(self.sample_array1))[:n_added]].int()
        # Remove 0
        selected_degree_distribution = torch.where(selected_degree_distribution > 0, selected_degree_distribution, self.mean_degree // 2)
        start = time()
        with tqdm(total=n_added) as pbar:
            for added_node in range(n_added):
                controller.homophily_decrease_score = self.homophily_decrease_score.clone()
                if selected_degree_distribution[added_node] > 2 * self.mean_degree:
                    selected_degree_distribution[added_node] = int(2 * self.mean_degree)
                pbar.set_description(f"Attack injected node with ID {added_node}")
                added_node_label = torch.tensor(np.random.choice(self.classes, 1)[0], device=self.device)
                self.injected_nodes_origin = torch.cat((self.injected_nodes_origin, added_node_label.unsqueeze(0)))
                self.injected_nodes_classes = torch.cat((self.injected_nodes_classes, added_node_label.unsqueeze(0)))
                added_node_feature = self.make_statistic_features(added_node, n_added=1, n_added_labels=added_node_label.unsqueeze(0))
                modified_adj = adjReshapeAddDim(self.modified_adj, self.nnodes + 1, self.device)
                modified_features = torch.vstack((self.modified_features, added_node_feature))
                first_potential_edges = self.get_potential_edges(added_node_label)
                pop_size = len(first_potential_edges)
                self.n_added_labels = torch.hstack((self.labels, self.injected_nodes_classes))
                evaluator = SGAEvaluator(feats=modified_features, adj=modified_adj, test_index=data_loader.test_index, labels=self.n_added_labels, pop_size=pop_size, device=self.device)
                evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator, W=self.W, edge_list=first_potential_edges)
                edges_ranks_score = evaluator(torch.randperm(len(first_potential_edges), device=self.device).int())
                edges_ranks = first_potential_edges
                edges_ranks_zip = zip(edges_ranks_score, self.homophily_decrease_score[edges_ranks[:, 1]], edges_ranks)
                edges_ranks_zip = sorted(edges_ranks_zip, key=lambda edges_ranks_zip: (edges_ranks_zip[0], -edges_ranks_zip[1]))
                edges_ranks_scores_list = list(zip(*edges_ranks_zip))
                edges_ranks = edges_ranks_scores_list[2]
                final_potential_edges = torch.stack([edges_ranks[i] for i in range(int(len(edges_ranks) * self.ratio))])
                best_single_link_loss = edges_ranks_scores_list[0][0]
                if selected_degree_distribution[added_node] != 1:
                    self.n_added_labels = torch.hstack((self.labels, self.injected_nodes_classes))
                    evaluator = SGAEvaluator(feats=modified_features, adj=modified_adj, test_index=data_loader.test_index, labels=self.n_added_labels, pop_size=pop_size, device=self.device)
                    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator, W=self.W, edge_list=final_potential_edges)
                    if controller.mode == "s" or controller.mode == "sm":
                        elite_edge, elite_edge_score = controller.calculate(max_generation=max_generation, evaluator=evaluator)
                    elif controller.mode == "m" or controller.mode == "mnm":
                        with Manager() as manager:
                            return_dict = manager.dict()
                            mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size, return_dict), nprocs=world_size, join=True)
                            elite_edge, elite_edge_score = [], []
                            for key, value in return_dict.items():
                                elite_edge.append(return_dict[key][0])
                                elite_edge_score.append(return_dict[key][1])
                            top_index = elite_edge_score.index(min(elite_edge_score))
                            elite_edge = elite_edge[top_index].to(self.device)
                            elite_edge_score = elite_edge_score[top_index].to(self.device)
                    else:
                        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")
                else:  # if only need to attack 1 edge, we can directly use the output of single-link attacks and do not need to employ GA
                    elite_edge = final_potential_edges[0]
                    elite_edge_score = best_single_link_loss
                elite_edge = elite_edge.flatten().int()
                # obtain the final adj
                modified_adj = self.get_modified_adj_by_edges_ranks(modified_adj.to_dense(), elite_edge_score, elite_edge, verbose=False)
                tag_id = 1
                tmp = self.labels.clone()
                tmp = torch.cat((tmp, self.injected_nodes_origin.clone()))
                while tag_id < len(elite_edge):
                    current_neighbors = torch.where(modified_adj[elite_edge[tag_id]] == 1)[0]
                    current_neighbors = current_neighbors[current_neighbors != self.nnodes]
                    current_neighbors_len = len(torch.where(modified_adj[elite_edge[tag_id]] == 1)[0])
                    current_samelabel_neighbors_len = len(torch.where(tmp[current_neighbors] == tmp[elite_edge[tag_id]])[0])
                    if current_neighbors_len == 0:
                        current_neighbors_len = 1
                    self.homophily_score[elite_edge[tag_id]] = current_samelabel_neighbors_len / (current_neighbors_len)
                    self.homophily_decrease_score[elite_edge[tag_id]] = current_samelabel_neighbors_len / ((current_neighbors_len + 1) * (current_neighbors_len + 1) + current_neighbors_len + 1)
                    tag_id += 2
                self.homophily_index = torch.tensor(np.argsort(-self.homophily_decrease_score.cpu().numpy()), device=self.device)

                # inject the current node to the original adj and f matrices
                self.modified_features = modified_features
                self.modified_adj = modified_adj.to_sparse_coo()
                self.nnodes = self.nnodes + 1
                attack_acc, asr = self.test()
                best_acc.append(attack_acc)
                best_asr.append(asr)
                end = time()
                time_list.append(end-start)
                pbar.set_postfix(Loss=elite_edge_score.item(), Acc=min(best_acc), ASR=max(best_asr))
                pbar.update(1)
        print('\nFinish attacks\n')
        top_index = best_acc.index(min(best_acc))
        print(f"Min acc: {best_acc[top_index]} -> Asr: {best_asr[top_index]}")
        print(f"Best acc: {best_acc[-1]} -> Asr: {best_asr[-1]}")
        self.save(controller, self.dataset, [torch.tensor([0, 0])], [best_acc[top_index], best_asr[top_index], time_list[-1]], time_list, method="SGA", bestAcc=best_acc, bestASR=best_asr)

    def save(self, controller: SGAController, dataset, gene, best_metric, time_list, method, **kwargs):
        save_path = controller.path + dataset + '_crossover_rate_' + str(controller.crossover_rate) + '_mutate_rate_' + str(controller.mutate_rate) + f'_{method}.txt'
        with open(save_path, 'a+') as f:
            f.write(current_time())
            f.write(f"\nCurrent mode: {controller.mode}. Current pop_size: {controller.pop_size}\n")
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestAcc']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestASR']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i.tolist() for i in gene]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(time_list) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(best_metric) + '\n')

    def test(self, surrogate=False):
        if surrogate:
            # ori_acc = Acc(self.features, self.adj, self.labels, self.idx_test, self.surrogate)
            attack_acc = Acc(self.modified_features, self.modified_adj, self.n_added_labels.long(), self.idx_test, self.surrogate)
            asr = AS_Rate(self.features, self.modified_features, self.adj, self.modified_adj, self.idx_test, self.surrogate)
        else:
            # ori_acc = Acc(self.features, self.adj, self.labels, self.idx_test, self.classifier)
            attack_acc = Acc(self.modified_features, self.modified_adj, self.n_added_labels.long(), self.idx_test, self.classifier)
            asr = AS_Rate(self.features, self.modified_features, self.adj, self.modified_adj, self.idx_test, self.classifier)
        return attack_acc, asr

    def get_linearized_weight(self):
        W = self.surrogate.gc1.weight @ self.surrogate.gc2.weight
        return W.detach()

    def get_sorted_features(self, zeroone_features):
        MAX_N = 100
        features_id = []
        feature_avg = []
        for label in self.classes:
            label_features = self.features[self.labels == label]
            zeroone_label_features = zeroone_features[self.labels == label]
            count = zeroone_label_features.sum(0)
            real_count = label_features.sum(0)
            count[count == 0] = 1
            current_avg = real_count / count
            df = pd.DataFrame(columns=['count', 'features_id'], data={'count': count[count.nonzero(as_tuple=True)[0]].cpu(), 'features_id': count.nonzero(as_tuple=True)[0].cpu()})
            df = df.sort_values('count', ascending=False)
            df.name = 'Label ' + str(label)
            features_id.append(df['features_id'].values[:MAX_N])
            feature_avg.append(current_avg)
        self.features_id = torch.tensor(np.array(features_id), device=self.device)
        self.feature_avg = feature_avg

    def make_statistic_features(self, added_node, n_added, n_added_labels=None):
        self.injected_nodes_classes[added_node] = n_added_labels[0]
        n_added_features = torch.zeros((n_added, self.features_dim), device=self.device)
        for i in range(n_added):
            n_added_features[0][self.major_features_candidates[n_added_labels[0]]] = self.feature_avg[0][self.major_features_candidates[n_added_labels[0]]]
        return n_added_features.to_sparse_coo()

    def get_potential_edges(self, added_node_label):
        new_candadite = []
        [new_candadite.append(i.cpu()) for i in self.homophily_index.clone() if not i in torch.where(self.labels == added_node_label)[0]]
        new_candadite = np.array(new_candadite)
        size = int(len(new_candadite))
        return torch.tensor(np.column_stack((np.tile(self.nnodes, size), new_candadite[:size])), dtype=torch.int, device=self.device)

    def get_modified_adj_by_edges_ranks(self, modified_adj, scores, edges, verbose=True):
        ori_node = edges[0].int()
        n_added_node = edges[1::2].int()
        self.ori_nodes = torch.cat((self.ori_nodes, n_added_node))
        modified_adj[ori_node, n_added_node] = 1
        modified_adj[n_added_node, ori_node] = 1
        if verbose:
            print("Edge perturbation: {} , loss: {}".format(edges, scores))
        return modified_adj


def SGA(mode, max_generation, data_loader, controller: SGAController, surrogate, classifier, homophily_ratio, world_size):
    controller.mode = mode
    # mapping matrix to handle continuous feature
    zeroone_features = data_loader.feats.clone().to_dense()
    zeroone_features[zeroone_features > 0] = 1
    algorithm = SGAAlgorithm(
        data_loader=data_loader,
        surrogate=surrogate,
        classifier=classifier,
        idx_test=data_loader.test_index,
        homophily_ratio=homophily_ratio,
        zeroone_features=zeroone_features,
        device=controller.device
    )
    algorithm.main(data_loader=data_loader, controller=controller, max_generation=max_generation, world_size=world_size)
