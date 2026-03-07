import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evox import workflows, monitors
from evox import Problem, jit_class, Algorithm, State
import jax
import jax.numpy as jnp
from jax import random, jit
import networkx as nx
from tqdm import tqdm
from absolute_path import dataset_path
from time import time

# karate dolphins football email-Eu-core
data_set = 'football'
if data_set == "dolphins" or data_set == "football" or data_set == "email-Eu-core":
    G = nx.read_gml(os.path.join(dataset_path, data_set, data_set + '.gml'), label="id")
else:
    G = nx.read_adjlist(os.path.join(dataset_path, data_set + '.txt'), nodetype=int)
A = jnp.array(nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))))
nodes_num = len(A)
edge_number = len(G.edges())
edge_list = jnp.array(list(G.edges()))
k = int(0.1 * nodes_num)
pop_size = 100
Eye = jnp.eye(nodes_num)
ONE = jnp.ones((pop_size, k), dtype=jnp.float32)
ONE_distance = jnp.ones((nodes_num, nodes_num), dtype=jnp.float32)
ONE_distance = ONE_distance.at[jnp.diag_indices(nodes_num)].set(0.0)

D_inverse = jnp.diag(jnp.float_power(jnp.sum(A + Eye, axis=1), -1))
normalized_matrix = jnp.dot(D_inverse, A)

approximate_deepwalk_matrix = jnp.multiply(1 / 2, jnp.add(normalized_matrix, jnp.dot(normalized_matrix, normalized_matrix)))

u, s, v = jnp.linalg.svd(approximate_deepwalk_matrix, full_matrices=False)

embedding = jnp.matmul(u, jnp.diag(jnp.sqrt(s)))
E_dots = jnp.sum(jnp.matmul(embedding, embedding), axis=1).reshape(nodes_num, 1)

distance_org = jnp.multiply(
    jnp.float_power(
        jnp.absolute(
            jnp.add(
                jnp.dot(E_dots, jnp.ones(shape=((1, nodes_num)))),
                jnp.dot(jnp.ones(shape=((nodes_num, 1))), E_dots.T)
            ) - 2 * jnp.dot(embedding, embedding.T)),
        1 / 2),
    ONE_distance)


@jit
def cal_svd(approximate_deepwalk_matrix):
    u, s, v = jnp.linalg.svd(approximate_deepwalk_matrix, full_matrices=False)
    return u, s, v


@jit
def cal_e_dot(embedding):
     return jnp.sum(jnp.matmul(embedding, embedding), axis=1).reshape(nodes_num, 1)


@jit
def compute_adjacency_matrix(population_idx, edge_list):
    del_edge_list = edge_list[population_idx]
    adj_matrix = A.at[del_edge_list[:, 0], del_edge_list[:, 1]].set(0)
    return adj_matrix


@jit
def calculate_embedding_and_distance(adj_matrix, ONE_distance):
    normalized_matrix = jnp.matmul(
        jnp.diag(jnp.float_power(jnp.sum(adj_matrix + Eye, axis=1), -1)),
        adj_matrix
    )  # D^(-1)@A

    approximate_deepwalk_matrix = jnp.multiply(
        1 / 2,
        jnp.add(
            normalized_matrix,
            jnp.matmul(normalized_matrix, normalized_matrix)
        )
    )

    u, s, v = cal_svd(approximate_deepwalk_matrix)
    embedding = jnp.matmul(u, jnp.diag(jnp.sqrt(s)))
    E_dots = cal_e_dot(embedding)
    distance = jnp.matmul(
        jnp.float_power(
            jnp.abs(
                jnp.add(
                    jnp.matmul(E_dots, jnp.ones((1, nodes_num))),
                    jnp.matmul(jnp.ones((nodes_num, 1)), E_dots.T)
                ) - 2 * jnp.matmul(embedding, embedding.T)
            ), 1 / 2),
        ONE_distance
    )
    return distance


@jit
def fitness_function(distance, distance_org):
    fitness_one = 1 - abs(
        jnp.corrcoef(
            jnp.vstack((distance.flatten(), distance_org.flatten()))
        )[0, 1]
    )
    return fitness_one


@jit_class
class OneMax(Problem):
    def __init__(self, adj, pop_size, edge_list, graph) -> None:
        super().__init__()
        self.adj = adj.copy()
        self.pop_size = pop_size
        self.edge_list = edge_list
        self.graph = graph.copy()

    def evaluate(self, state, population):
        population = population.astype(jnp.int32)
        def compute_individual(population_idx):
            adj_matrix = compute_adjacency_matrix(population_idx, self.edge_list)
            distance = calculate_embedding_and_distance(adj_matrix, ONE_distance)
            fitness_one = fitness_function(distance, distance_org)
            return fitness_one
        fitness = jax.vmap(compute_individual)(population)
        return fitness, state


from jax import lax
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
    def __init__(self, pop_size, budget, enum, pc, pm):
        super().__init__()
        self.pop_size = pop_size
        self.budget = budget
        self.enum = enum
        self.pc = pc
        self.pm = pm

    def setup(self, key):
        key, subkey = random.split(key)
        pop = random.choice(key=subkey, a=self.enum, shape=(self.pop_size, self.budget), replace=True).astype(jnp.float32)
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

    def ask(self, state):
        key, shuffle_key, cm_key, mm_key, mp_key = random.split(state.key, 5)

        # shuffle population for crossover
        new_population1 = state.pop.copy()
        new_population2 = state.pop.copy()
        random.permutation(key=shuffle_key, x=new_population2, independent=True)

        # crossover operation
        crossover_matrix = random.choice(key=cm_key, a=jnp.array([0, 1]), shape=(self.pop_size, self.budget),
                                         p=jnp.array([1 - self.pc, self.pc]))
        crossover_population = new_population1 * (ONE - crossover_matrix) + new_population2 * crossover_matrix

        # mutation operation
        mutation_matrix = random.choice(key=mm_key, a=jnp.array([0, 1]), shape=(self.pop_size, self.budget),
                                        p=jnp.array([1 - self.pm, self.pm]))
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

    def tell(self, state, fitness):
        stack_population = jnp.vstack((state.pop, state.offsprings))
        stack_fitness_list = jnp.hstack((state.fit, fitness))

        top_index = jnp.argsort(stack_fitness_list)[:self.pop_size]

        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index]
        return state.update(pop=population, fit=fitness_list)


start = time()
algorithm = CustomGA(
    pop_size=pop_size,
    budget=k,
    enum=edge_number,
    pc=0.5,
    pm=0.3
)
problem = OneMax(adj=A, pop_size=pop_size, edge_list=edge_list, graph=G)
monitor = monitors.EvalMonitor()

workflow = workflows.StdWorkflow(
    algorithm,
    problem,
    monitors=[monitor],
    opt_direction="max",
)

key = random.PRNGKey(42)
state = workflow.init(key)

print(monitor.get_best_fitness())
with tqdm(total=1500) as pbar:
    for i in range(1500):
        state = workflow.step(state)
        fitness = monitor.get_best_fitness()
        pbar.update(1)
        pbar.set_postfix(fitness=fitness)

end = time()
print(end-start)
