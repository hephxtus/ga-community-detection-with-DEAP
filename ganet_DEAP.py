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
import networkx.algorithms.community as nx_comm

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



def community_detection(graph,population=300,generation=30,r=1.5):
    """
    :param nodes: number of nodes in the network
    :param edges: number of edges in the network
    :param population: number of individuals in the population
    :param generation: number of generations
    :param r: crossover rate
    :return:
    """
    CXPB, MUTPB = 0.8, 0.2
    # create the graph
    # graph=nx.Graph()
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    Adj = nx.adjacency_matrix(graph)
    nodes_length = len(graph.nodes())

    # print(nx.louvain_partitions(graph))
    toolbox.register("chromosomes", generate_chrom, nodes_length)
    toolbox.register("subsets", find_subsets, toolbox.chromosomes())
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.chromosomes, nodes_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", community_score, r=r, Adj=Adj)
    toolbox.register("evaluate", nx_comm.modularity, G=graph, weight='weight', resolution=1.0)

    pop = toolbox.population(n=population)
    # toolbox.register("deme", tools.initRepeat, list, toolbox.individual)
    #
    # DEME_SIZES = 10, 50, 100
    # population = [toolbox.deme(n=i) for i in DEME_SIZES]
    ind = toolbox.individual()
    # print(ind)
    # print(ind)
    # print(toolbox.evaluate(ind, find_subsets(ind)))
    # ind.fitness.values = toolbox.evaluate(ind, find_subsets(ind))
    # print(ind.fitness.valid)
    # print(pop)

    for ind in pop:
        # print(ind)
        ind.subset = find_subsets(ind, G=graph)
        # ind.subset = graph.subgraph(ind)
        # print(ind.subset)
        # i = toolbox.evaluate(communities=ind.subset)
        # print(i)
        ind.fitness.values = (toolbox.evaluate(communities=ind.subset),)


    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selRoulette, fit_attr="fitness")
    # evaluate the population
    # print("  Evaluated %i individuals" % len(pop))

    # do the evolution
    for g in range(generation):
        size = int(np.floor(population * 0.9))
        pop.sort(key=lambda x: x.fitness, reverse=True)
        offspring = toolbox.select(pop, size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            # print(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values
            # cross two individuals with probability CXPB
        # for mutant in offspring:
        #     toolbox.mutate(mutant)
        #     del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.subset = find_subsets(ind, graph)
            ind.fitness.values = (toolbox.evaluate(communities=ind.subset),)

        # print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

    # print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1, fit_attr="fitness")[0]
    print("Best individual is %s, %s" % (best_ind.subset, best_ind.fitness.values))

    unique_coms = np.unique(list(best_ind))
    cmap = {
        0: 'maroon',
        1: 'teal',
        2: 'black',
        3: 'orange',
        4: 'green',
        5: 'yellow'
    }

    node_cmap = [cmap[i] for i in range(len(best_ind.subset))]
    print(node_cmap)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=75, alpha=0.8, node_color=list(node_cmap), with_labels=True)
    plt.show()


        # offspring.fitness.values = toolbox.evaluate(offspring, find_subsets(offspring))
        # pop[:] = toolbox.select(pop + [offspring])
        # print("  Evaluated %i individuals" % len(pop))
        # fits = [ind.fitness.values[0] for ind in pop]
        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x * x for x in fits)
        # std = abs(sum2 / length - mean ** 2) ** 0.5
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)



def generate_chrom(nodes_length):
    rand = np.random.randint(0, nodes_length)
    return rand

def merge_subsets(sub, to_skip):
    arr = []
    to_skip = []

    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            # print(new)
            for x in sub:
                if sub[s] & x and x not in to_skip:
                    new = new | x
                    print("x:", x)
                    to_skip.append(x)
            # print("new:",new)
            arr.append(new)
    # print("to skip:", to_skip)
    return arr, to_skip


def find_subsets(chrom, G: nx.Graph):
    """
    Finds all subsets of a given chromsome

    :param chrom:
    :return:
    """
    temp = set(x for x in range(len(chrom)))
    subs = []
    to_skip = set()
    sub = []
    for x in range(len(chrom)):
        if chrom[x] != x:
            sub.append((x, chrom[x]))
        else:
            sub.append(tuple(random.choice(list(G.edges(x)))))
    # sub = [(x, chrom[x]) for x in range(len(chrom))]
    # print(sub)
    # print(chrom)
    for n in chrom:
        if n in to_skip:
            continue
        connections = G.edges(n)
        new = set()
        # print(to_skip)
        for e in sub:
            if e in connections and (e[1] not in to_skip or e[1] == n) and (e[0] not in to_skip or e[0] == n):
                # print(e)
                new = new | set(e)
                to_skip = to_skip | set(e)
        # print(new)
        # print("----")
        if new != set():
            subs.append(new)
            temp= temp - new
    # print("remaining:", temp) if temp != set() else print("no remaining")
    if temp != set():
        subs.append(temp)
    return subs
    # for t in temp:
    #     # print(sub[t][0], sub[t][1])
    #     for i in range(len(subs)):
    #         s = subs[i]
    #         if sub[t][1] in s:
    #             s = s | {sub[t][0]}
    #             # print("adding:", sub[t][1], "to", s)
    #             subs[i] = s

    return subs
        # new = set()
        # for s in sub:
        #     if s in G.edges(n) and s not in to_skip:
        #         new = new | set(s)
        #         to_skip.append(s)
        # if new != set():
        #     subs.append(new)

    # print(subs)
    # print(list(G.edges(chrom[0])))
    # print([{x, chrom[x]} for x in range(len(chrom))])
    #
    # result = sub
    # i = 0
    # to_skip = []
    # while i < len(result):
    #     print("i:", i)
    #     print(result)
    #     candidate, to_skip = merge_subsets(result, to_skip)
    #     if candidate != result:
    #         result = candidate
    #     else:
    #         break
    #     result = candidate
    #     i += 1
    # print(result)
    # return result

def getModularity(chrom, network, subsets):
    Q = 0
    G = network.copy()
    nx.set_edge_atAtributes(G, {e: 1 for e in G.edges}, 'weight')
    A = nx.to_scipy_sparse_matrix(G).astype(float)
    # for undirected graphs, in and out treated as the same thing
    out_degree = in_degree = dict(nx.degree(G))
    M = 2. * (G.number_of_edges())
    print("Calculating modularity for undirected graph")

    nodes = list(G)
    Q = np.sum([A[i, j] - in_degree[nodes[i]] * \
                out_degree[nodes[j]] / M \
                for i, j in product(range(len(nodes)), range(len(nodes)))
                if subsets[nodes[i]] == subsets[nodes[j]]])
    return Q / M

def community_score(chrom,subsets,r,Adj):
    """
    :param chrom: chromosome of the individual
    :param subsets: connected components of the graph
    :param r: crossover rate
    :param Adj: adjacency matrix of the graph
    :return:
    """
    matrix = Adj.toarray()
    CS=0
    # print(subsets)
    for s in subsets:
        submatrix = np.zeros((len(chrom),len(chrom)),dtype=int)
        for i in s:
            for j in s:
                # print(s)
                submatrix[i][j]=matrix[i][j]
        M=0
        v=0
        for row in list(s):
            row_mean = np.sum(submatrix[row])/len(s)
            v+=np.sum(submatrix[row])
            M+=(row_mean**r)/len(s)
        CS+=M*v
    return CS,

def mutation(chrom,Adj,mutation_rate):
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
            mutant = np.random.randint(1, len(chrom))
            # print(Adj[mutant])
            row = Adj[mutant].toarray()[0]
            # print(row)
            neighbor = [chrom[i] for i in range(len(row)) if row[i] == 1]
            # print(chrom)
            # print(mutant)
            if chrom[mutant] in neighbor:
                # print("yes")
                # print(neighbor, chrom[mutant])
                neighbor.remove(chrom[mutant])
                to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
                chrom[mutant] = neighbor[to_change]
                neighbor.append(chrom[mutant])
    return chrom

# n = 10
# G = generate_network(n)
# print(nx.info(G))

# visualize graph


# nodes = [0,1,2,3,4,5,6,7,8,9,10]
# edges = [(0, 1), (0, 4), (1, 2), (2, 3), (1, 3), (3, 0), (0, 2), (4, 5), (5, 6), (6, 7), (10, 8), (10, 9), (8, 9),
#          (8, 7), (9, 7), (7, 10)]
graph = nx.complete_graph(20)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, node_size=75, alpha=0.8)
plt.show()
nodes = graph.nodes
edges = graph.edges
i = 0
start = time.time()
try:
    while i < 10:
        interval = time.time()
        community_detection(graph)
        print("Time taken: ", time.time()-interval)
        i += 1
    end = time.time()
    print("average: %d" % ((end-start)/i))
except KeyboardInterrupt:
    end = time.time()
    print("average: %d" % ((end - start) / i))