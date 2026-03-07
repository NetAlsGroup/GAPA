import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evox import algorithms, problems, workflows, monitors
from evox import Problem, jit_class, Algorithm, State
from evox.operators import selection
import jax
import jax.numpy as jnp
from jax import random, jit
import networkx as nx
from time import time
from tqdm import tqdm
from jax import lax
from absolute_path import dataset_path
import numpy as np
from collections import Counter

"""
    Set hyperparameters
    Generate the initial population
    Do
        Generate Offspring
            Selection
            Crossover
            Mutation
        Compute fitness
        Replace the population
    Until stopping criterion
    
        |
        v
        
    Set hyperparameters # __init__
    Generate the initial population # setup
    Do
        # ask
        Generate Offspring
            Mating Selection
            Crossover
            Mutation
    
        # problem.evaluate (not part of the algorithm)
        Compute fitness
    
        # tell
        Survivor Selection
    Until stopping criterion
"""


start = time()
# Circuit A01 humanDiseasome yeast1
data_set = 'yeast1'
if data_set == "dolphins" or data_set == "football" or data_set == "email-Eu-core":
    G = nx.read_gml(os.path.join(dataset_path, data_set, data_set + '.gml'), label="id")
else:
    G = nx.read_adjlist(os.path.join(dataset_path, data_set + '.txt'), nodetype=int)
A = jnp.array(nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))))
nodes = jnp.array(list(G.nodes()))
nodes_num = len(A)
edge_number = len(G.edges())
edge_list = jnp.array(list(G.edges()))
k = int(0.1 * nodes_num)
selected_genes_rate = 0.4
selected_genes_num = int(nodes_num * selected_genes_rate)
pop_size = 80
Eye = jnp.eye(nodes_num)
ONE = jnp.ones((pop_size, k), dtype=jnp.float32)


def CNDTest(graph, critical_nodes):
    copy_g = graph.copy()
    for node in critical_nodes:
        try:
            copy_g.remove_node(node.item())
        except:
            pass
    total = 0
    sub_graph_list = list(nx.connected_components(copy_g))
    for sub_graph_i in range(len(sub_graph_list)):
        total += len(sub_graph_list[sub_graph_i]) * (len(sub_graph_list[sub_graph_i]) - 1) / 2
    return total


def pop_greedy_cutoff(graph, genes, pop_num):
    greedy_indi = []
    for _i in range(pop_num):
        temp_fitness = []
        temp_population = []
        # temp_population = self._remove_repeat(temp_population)
        for j in range(pop_size):
            temp_key = random.PRNGKey(np.random.randint(0, 10000))
            temp_genes = random.choice(key=temp_key, a=nodes_num, shape=(k,), replace=True)
            temp_population.append(temp_genes)
            temp_fitness.append(CNDTest(graph, genes[temp_genes]))
        top_index = jnp.array(temp_fitness).argsort()[:1]
        greedy_indi.append(temp_population[top_index.item()])

    genes = []
    for pop in greedy_indi:
        genes = genes + pop.tolist()
    data = dict(Counter(genes))
    data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(selected_genes_num / 2)]
    genes = jnp.array([data[i][0] for i in range(len(data))])
    # genes = list(set(genes + list(history[0])))
    genes = nodes[genes].tolist()
    degree = nx.degree_centrality(graph)
    degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
    degree = [degree[i][0] for i in range(len(degree))]
    count_i = 0
    while len(genes) < selected_genes_num:
        if degree[count_i] not in genes:
            genes.append(degree[count_i])
        count_i = count_i + 1
    return jnp.array(genes)


@jit
def update_one_pop(one_pop):
    adj = A.at[nodes[one_pop]].set(0)
    adj = adj.T.at[nodes[one_pop]].set(0).T
    adj = matrix_matmul(adj)
    return jnp.max(adj)


@jit
def matrix_matmul(adj: jnp.ndarray):
    adv_A = jnp.matmul(adj.astype(jnp.float32), adj.T.astype(jnp.float32))
    M2 = jnp.matmul(adv_A + Eye, adv_A + Eye)
    M4 = jnp.matmul(M2, M2)
    M6 = jnp.matmul(M4, M2)
    return jnp.count_nonzero(M6, axis=1)


@jit_class
class OneMax(Problem):
    def __init__(self) -> None:
        super().__init__()

    # @jit
    def evaluate(self, state, population):
        population = population.astype(jnp.int32)
        fit = jax.vmap(update_one_pop)(population).astype(jnp.float32)
        return fit, state


@jit
def selection(population, fitness_list, rand_key):
    copy_pop = population.copy()

    fitness_list = jnp.maximum(fitness_list, 0)
    fitness_sum = fitness_list.sum()

    normalize_fit = lax.cond(
        fitness_sum > 0,
        lambda _: fitness_list / fitness_sum,
        lambda _: jnp.ones_like(fitness_list) / len(fitness_list),
        operand=None
    )

    samples = random.choice(
        key=rand_key,
        a=len(copy_pop),
        p=normalize_fit,
        shape=(len(copy_pop),),
        replace=True
    ).astype(jnp.int32)

    return copy_pop[samples]


@jit_class
class CustomGA(Algorithm):
    def __init__(self, pop_size, pm, pc, k):
        super().__init__()
        self.pop_size = pop_size
        self.budget = k
        # mutation rate and crossover rate
        self.pm = pm
        self.pc = pc

    def setup(self, key):
        key, subkey = random.split(key)
        pop = random.choice(key=subkey, a=nodes_num, shape=(self.pop_size, self.budget), replace=True).astype(jnp.float32)
        return State(
            pop=pop,
            offsprings=jnp.empty((self.pop_size * 2, self.budget)),
            fit=jnp.full((self.pop_size,), jnp.inf),
            key=key,
        )

    def init_ask(self, state: State):
        population = state.pop
        return population, state

    def init_tell(self, state: State, fitness: jax.Array):
        # update the best fitness list with current best fitness
        return state.update(fit=fitness)

    # @jit
    def ask(self, state):
        key, shuffle_key, cm_key, mm_key, mp_key, sel_key = random.split(state.key, 6)

        # shuffle population for crossover
        new_population1 = state.pop.copy()
        new_population2 = selection(state.pop.copy(), state.fit.copy(), sel_key)
        # random.permutation(key=shuffle_key, x=new_population2, independent=True)

        # crossover operation
        crossover_matrix = random.choice(key=cm_key, a=jnp.array([0, 1]), shape=(self.pop_size, self.budget), p=jnp.array([1 - self.pc, self.pc]))
        crossover_population = new_population1 * (ONE - crossover_matrix) + new_population2 * crossover_matrix

        # mutation operation
        mutation_matrix = random.choice(key=mm_key, a=jnp.array([0, 1]), shape=(self.pop_size, self.budget), p=jnp.array([1 - self.pm, self.pm]))
        mutated_values = random.choice(key=mp_key, a=nodes_num, shape=(self.pop_size, self.budget), replace=True)
        mutation_population = crossover_population * (ONE - mutation_matrix) + mutated_values * mutation_matrix
        mutation_population = mutation_population.astype(jnp.float32)

        # so the offspring is twice as large as the population
        offsprings = jnp.concatenate(
            (
                mutation_population,
                crossover_population
            ),
            axis=0,
        )
        # return the candidate solution and update the state
        return offsprings, state.update(offsprings=offsprings, key=key)

    # @jit
    def tell(self, state, fitness):
        stack_population = jnp.vstack((state.pop, state.offsprings))
        stack_fitness_list = jnp.hstack((state.fit, fitness))

        top_index = jnp.argsort(stack_fitness_list)[:self.pop_size]

        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index]
        return state.update(pop=population, fit=fitness_list)


nodes = pop_greedy_cutoff(G, nodes, pop_num=10)
nodes_num = len(nodes)

algorithm = CustomGA(
    pop_size=pop_size,
    pm=0.2,
    pc=0.6,
    k=k
)
problem = OneMax()
monitor = monitors.EvalMonitor()
workflow = workflows.StdWorkflow(algorithm, problem, monitors=[monitor], opt_direction="min")

key = random.PRNGKey(np.random.randint(0, 10000))
state = workflow.init(key)

with tqdm(total=5000) as pbar:
    pbar.set_description(f"Calculate on dataset: {data_set}...")
    for i in range(5000):
        state = workflow.step(state)
        fitness = monitor.get_best_fitness()
        pbar.set_postfix(fitness=fitness)
        pbar.update(1)

end = time()
print(end-start)
