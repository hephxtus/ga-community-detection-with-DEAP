import itertools
import pickle
import random
import sys
import time
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import networkx as nx
from deap import base, creator, tools
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
from cdlib import algorithms
import igraph
import leidenalg as la
import networkx.algorithms.community as nx_comm
from networkx import Graph

"""
implementation of the DEAP algorithm for community detection

Genetic representation: Our clustering algorithm uses locus-based adjacency representation. In this graph-based 
                        representation an individual of the population consists of N genes {g1,...,gN} and each gene can 
                        assume allele values jin the range {1,...,N}. Genes and alleles represent nodes of the graph 
                        G=(V, E) modelling a social network SN,and a value j assigned to the ith gene is interpreted as 
                        a link between the nodes i and j of V. This means that in the clustering solution found i and j 
                        will be in the same cluster. A decoding step, however, is necessary to identify all the 
                        components of the corresponding graph. The nodes participating to the same component are 
                        assigned to one cluster.

Objective Function: As described above, the decoding of an individual provides a different number k of components 
                    {S1,...Sk}in which the graph is partitioned. We are interested in identifying a partitioning that 
                    optimizes the community score because, as already discussed in the previous section, this guarantees 
                    highly intra-connected and sparsely inter-connected communities. The objective function is thus 
                    CS = Standard div (upper limit = k, lower limit = i) of Q(Si)

Initialization: Our initialization process takes in account the effective connections of the nodes in the social 
                network. A random generation of individuals could generate components that in the original graph are 
                disconnected. In fact, a randomly generated individual could contain an allele value j in the ith 
                position, but no connection exists between the two nodes iand j, i.e. the edge (i, j) is not present. 
                In such a case itis obvious that grouping in the same cluster both nodes iand j is a wrong choice. In 
                order to overcome this drawback, once an individual is generated, it is repaired, that is a check is 
                executed to verify that an effective link exists between a gene at position i and the allele value j. 
                This value is maintained only if the edge (i, j) exists. Otherwise, j is substituted with one of the 
                neighbors of i. This guided initialization biases the algorithm towards a decomposition of the network 
                in connected groups of nodes. We call an individual generating this kind of partitioning safe because it 
                avoids uninteresting divisions containing unconnected nodes. Safe individuals improve the convergence of 
                the method because the space of the possible solutions is restricted.

Uniform Crossover: We used uniform crossover because it guarantees the maintenance of the effective connections of the 
                    nodes in the social network in the child individual. In fact, because of the biased initialization, 
                    each individual in the population is safe,that is it has the property, that if a gene i contains a 
                    value j, then the edge (i, j) exists. Thus, given two safe parents, a random binary vector is 
                    created. Uniform crossover then selects the genes where the vector is a 1 from the ﬁrst parent, and 
                    the genes where the vector is a 0 from the second parent, and combines the genes to form the child. 
                    The child at each position i contains a value j coming from one of the two parents. 
                    Thus the edge (i, j) exists. This implies that from two safe parents a safe child is generated.

Mutation: The mutation operator that randomly change the value j of a i-th gene causes a useless exploration of the 
            search space, because of the same above observations on node connections. Thus the possible values an allele 
            can assume are restricted to the neighbors of gene i. This repaired mutation guarantees the generation of a 
            safe mutated child in which each node is linked only with one of its neighbors. Given a network SN and the 
            graph G modelling it, GA-Net starts with a population initialized at random and repaired to produce safe 
            individuals. Every individual generates a graph structure in which each component is a connected subgraph of 
            G.For a ﬁxed number of generations the genetic algorithm computes the ﬁtness function of each solution 
            member, and applies the specialized variation operators to produce the new population.
"""
# create the population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list)

# create the toolbox
toolbox = base.Toolbox()


def draw_communities(communities, G, title='TEST'):
    plt.clf()
    node_cmap = []
    cmap = {
        0: 'gray',
        1: 'tab:red',
        2: 'tab:blue',
        3: 'tab:orange',
        4: 'tab:green',
        5: 'yellow',
        6: 'tab:purple',
        7: 'tab:brown',
        8: 'tab:pink',
        9: 'tab:olive',
        10: 'tab:cyan',
        11: 'tab:gray',

    }
    communities = list(communities)
    print(communities)
    # print(list(map(lambda x: cmap[communities.index(x)], communities)))
    # nx.draw_networkx(G, node_color=list(map(lambda x: cmap[communities.index(x)], communities)), with_labels=True)
    for i, nodes in enumerate(communities):
        for node in nodes:
            node_cmap.append((node, {"color": cmap[i]}))
    print(node_cmap)

    modularity = nx_comm.modularity(G, communities)
    print(communities)
    for key in range(len(communities)):
        sub_graph = G.subgraph(communities[key])
        nx.draw_networkx(sub_graph, pos=nx.spring_layout(G), node_color=cmap[key], with_labels=True)
    plt.annotate(f'Modularity: {modularity}', xy=(0.5, 0.05), xycoords='figure fraction',
                 horizontalalignment='center', )
    plt.savefig(title)


def community_detection(pop, generation=30, population=100):
    """
    :param nodes: number of nodes in the network
    :param edges: number of edges in the network
    :param population: number of individuals in the population
    :param generation: number of generations
    :param r: crossover rate
    :return:
    """
    global convergence, convergence_max, convergence_min

    for ind in pop:
        ind.subset = toolbox.subsets(chrom=ind)
        ind.fitness.values = (toolbox.evaluate(communities=ind.subset),)

    pop.sort(key=lambda x: x.fitness, reverse=True)
    # do the evolution
    for g in range(generation):
        elites = pop[:int(population * .1)]
        size = int(np.floor(population * 0.9))
        # pop = pop[:size]
        offspring = toolbox.select(pop, size)
        offspring = list(map(toolbox.clone, offspring))
        # convergence.append(offspring[0].fitness.values[0])
        x = 0
        temppop = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1=child1, ind2=child2)
            toolbox.mutate(chrom=child1)
            toolbox.mutate(chrom=child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values
            # cross two individuals with probability CXPB
            # for mutant in offspring:

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.subset = toolbox.subsets(chrom=ind)
            ind.fitness.values = (toolbox.evaluate(communities=ind.subset),)

        offspring += elites
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        pop.sort(key=lambda x: x.fitness, reverse=True)
        convergence[g] = convergence[g] + pop[0].fitness.values[0]

    if max(convergence) < convergence_max:
        convergence_max = max(convergence)
    if min(convergence) > convergence_min:
        convergence_min = min(convergence)

    best_ind = tools.selBest(pop, 1, fit_attr="fitness")[0]

    # return best_ind.fitness.values
    # return be
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    return best_ind


def calculate_centrality(G):
    """
    use
    :param G: networkx graph
    :return: (node, centrality) pairings
    """
    centrality = nx.eigenvector_centrality(G)
    return centrality



def generate_chrom(nodes, Adj, centrality):
    chrom = np.array([], dtype=int)
    for x in nodes:
        rand = np.random.choice(nodes)
        while Adj[x, rand] < 1:
            rand = np.random.choice(nodes)
        chrom = np.append(chrom, rand)
    return creator.Individual(chrom)

def find_central_node(G, chrom):
    """
    Finds the node with the highest degree in the given chromsome

    :param chrom:
    :return:
    """
    degrees = G.degree(chrom)
    return max(degrees, key=lambda x: x[1])[0]

def merge_subsets(sub):
    arr = []
    to_skip = []
    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            for x in sub:
                if sub[s] & x:
                    new = new | x
                    to_skip.append(x)
            arr.append(new)
    return arr


def find_subsets(G, chrom, centrality):
    """
    Finds all subsets of a given chromsome

    :param chrom:
    :return:
    """
    temp = list(chrom[:])
    sub = []
    # while temp is not empty,
    # find the node with the highest centrality in temp
    # find all nodes immediately connected to it
    # add the set of nodes to sub
    # remove the nodes from temp
    # get only nodes in temp
    node_centralities = list(filter(lambda x: x[0] in temp, centrality))
    # while temp:
        # print(node_centralities)
        # print(temp)
        # # find the node with the highest centrality
        # node = max(node_centralities, key=lambda x: x[1])[0]
        # # remove the node from node centrality
        # node_centralities = list(filter(lambda x: x[0] != node, node_centralities))
        # # find all nodes immediately connected to it within temp
        # neighbors = list(G.subgraph(temp).neighbors(node))
        # # remove neighbours from node centralities
        # node_centralities = list(filter(lambda x: x[0] not in neighbors, node_centralities))
        # # add the set of nodes to sub
        # #add node to neighbors
        # neighbors.append(node)
        # sub.append(set(neighbors))
        # # remove the nodes from temp
        # temp = list(set(temp) - set(neighbors))
    for x in range(len(chrom)):
        neighbours = set(G.neighbors(x))
        intersect = list(neighbours.intersection(temp))
        if len(intersect) == 0 or (len(intersect) == 1 and {x, intersect[0]} in sub):
            sub.append({x, x})
        else:
            # print list of nodes in node_centralities
            # if node centrality does not intersect with intersect, print intersect
            if not any(map(lambda n: n[0] in intersect, node_centralities)):
                print("INTERSECT:", intersect)
            node = max(list(filter(lambda n: n[0] in intersect, node_centralities)), key=lambda n: n[1])[0]

            sub.append({x, node})
            # remove node from temp
            temp.remove(node)
            # remove node from node_centralities
            node_centralities = list(filter(lambda n: n[0] != node, node_centralities)) if node not in temp \
                else node_centralities

    result = sub
    candidate = merge_subsets(result)
    while candidate != result:
        result = candidate
        candidate = merge_subsets(result)
    return result





def find_closest_node(G, chrom):
    """
    Finds the node with the highest centrality by closeness in the given chromsome

    :param chrom:
    :return:
    """
    closeness = nx.closeness_centrality(G, list(chrom))
    return max(closeness, key=lambda x: x[1])[0]


def mutation(chrom, Adj, mutation_rate, G):
    """
    Mutation: The mutation operator that randomly change the value j of a i-th gene causes a useless exploration of the
            search space, because of the same above observations on node connections. Thus the possible values an allele
            can assume are restricted to the neighbors of gene i. This repaired mutation guarantees the generation of a
            safe mutated child in which each node is linked only with one of its neighbors. Given a network SN and the
            graph G modelling it, GA-Net starts with a population initialized at random and repaired to produce safe
            individuals. Every individual generates a graph structure in which each component is a connected subgraph of
            G.For a ﬁxed number of generations the genetic algorithm computes the ﬁtness function of each solution
            member, and applies the specialized variation operators to produce the new population.
    """

    if np.random.random_sample() < mutation_rate:
        chrom = chrom
        neighbor = []
        while len(neighbor) < 2:
            # find node with highest centrality value in chrom
            # mutant = max(chrom, key=lambda x: G.nodes[x]['centrality'])
            mutant = find_central_node(G, chrom)
            row = Adj[mutant].toarray()[0]
            neighbor = [i for i in range(len(row)) if row[i] > 0]

            if len(neighbor) > 1:
                neighbor.remove(chrom[mutant])
                to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
                chrom[mutant] = neighbor[to_change]
                neighbor.append(chrom[mutant])
    return chrom


# set random seed
np.random.seed(0)

# graph = nx.karate_club_graph()
graph = nx.read_gml('database/dolphins.gml')
labels = graph.nodes()
graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
pos = nx.spring_layout(graph)

# nx.draw_networkx(graph, pos, node_size=75, alpha=0.8)
# plt.show()
nodes = graph.nodes
edges = graph.edges
i = 0
max_iterations = 30

fits = []
times = []
while i < max_iterations:
    start = time.time()
    communities_louvain = list(nx_comm.label_propagation_communities(graph))
    fits.append(nx_comm.modularity(graph, communities_louvain))
    times.append(time.time() - start)
    i += 1
mean = float(sum(fits) / len(fits))
sum2 = sum([x * x for x in fits])
std = abs(sum2 / len(graph.nodes) - mean ** 2) ** 0.5
print("label propogation min", min(fits))
print("label propogation max", (max(fits)))
print("label propogation mean:", mean)
print("label propogation std:", std)
print("label propogation", "Time taken: ", sum(times) / len(times))

fits = []
times = []
i = 0

while i < max_iterations:
    start = time.time()
    communities_louvain = nx_comm.louvain_communities(graph)
    fits.append(nx_comm.modularity(graph, communities_louvain))
    times.append(time.time() - start)
    i += 1
mean = float(sum(fits) / len(fits))
sum2 = sum([x * x for x in fits])
std = abs(sum2 / len(graph.nodes) - mean ** 2) ** 0.5
print("louvain min", min(fits))
print("louvain max", (max(fits)))
print("louvain mean:", mean)
print("louvain std:", std)
print("louvain", "Time taken: ", sum(times) / len(times))

fits = []
times = []
i = 0
while i < max_iterations:
    start = time.time()
    communities_louvain = nx_comm.greedy_modularity_communities(graph)
    fits.append(nx_comm.modularity(graph, communities_louvain))
    times.append(time.time() - start)
    i += 1
mean = float(sum(fits) / len(fits))
sum2 = sum([x * x for x in fits])
std = abs(sum2 / len(graph.nodes) - mean ** 2) ** 0.5
print("greedy modularity min", min(fits))
print("greedy modularity max", (max(fits)))
print("greedy modularity mean:", mean)
print("greedy modularity std:", std)
print("greedy modularity", "Time taken: ", sum(times) / len(times))

fits = []
times = []
i = 0
while i < max_iterations:
    """implement leiden community detection"""
    start = time.time()
    communities_leiden = algorithms.leiden(graph)
    # print(communities_leiden.communities)
    fits.append(nx_comm.modularity(graph, communities_leiden.communities))
    times.append(time.time() - start)
    i += 1
mean = float(sum(fits) / len(fits))
sum2 = sum([x * x for x in fits])
std = abs(sum2 / len(graph.nodes) - mean ** 2) ** 0.5
print("leiden min", min(fits))
print("leiden max", (max(fits)))
print("leiden mean:", mean)
print("leiden std:", std)
print("leiden", "Time taken: ", sum(times) / len(times))

fits = []
times = []
i = 0
start = time.time()

fig = plt.figure()
plt.xlabel('Generation')
plt.ylabel('Modularity')
plt.title('Modularity Convergence')
convergence_max, convergence_min = 1, 0

CXPB, MUTPB = 0.8, 0.2
Adj = nx.adjacency_matrix(graph)
nodes = graph.nodes()
nodes_len = len(nodes)
print(nodes_len)
centrality = calculate_centrality(graph).items()
toolbox.register("individual", generate_chrom, nodes=nodes, Adj=Adj, centrality=centrality)
toolbox.register("subsets", find_subsets, G=graph, centrality=centrality)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=100)
toolbox.register("evaluate", nx_comm.modularity, G=graph, weight='weight', resolution=1.0)

toolbox.register("mate", tools.cxUniform, indpb=CXPB)
toolbox.register("mutate", mutation, Adj=Adj, mutation_rate=MUTPB, G=graph)
toolbox.register("select", tools.selRoulette, fit_attr="fitness")

# pickle.dump(toolbox, open("toolbox.p", "wb"))
# toolbox = pickle.load(open("toolbox.p", "rb"))
convergence = [0] * 30
try:
    while i < max_iterations:
        i += 1
        interval = time.time()
        pop = toolbox.population()
        scores = community_detection(pop)
        print("SCORES:", scores)
        fits.append(scores.fitness.values[0])
        # print("Time taken: ", time.time()-interval)
    end = time.time()
except KeyboardInterrupt:
    end = time.time()
finally:
    print(fits)

    print("average: %d" % ((end - start) / i))
    mean = float(sum(fits) / len(fits))
    sum2 = sum([x * x for x in fits])
    std = abs(sum2 / len(graph.nodes) - mean ** 2) ** 0.5
    print("GADeap min", min(fits))
    print("GADeap max", (max(fits)))
    print("GADeap mean:", mean)
    print("GADeap std:", std)
    print("final convergence=", convergence)
    # plt.ylim(convergence_max+0.01, convergence_min-0.01)
    for c, _ in enumerate(convergence):
        convergence[c] = convergence[c] / i

    plt.plot(convergence, zorder=i)
    plt.show()
    draw_communities(communities=scores.subset, G=graph)

# check degree and see if can be moved
