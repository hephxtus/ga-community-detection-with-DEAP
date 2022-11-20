import itertools
import random
import sys
import time
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import networkx as nx
from deap import base, creator, tools
import matplotlib.pyplot as plt
from cdlib import algorithms
import igraph
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





def community_detection(pop, graph, generation=30, population=100):
    """
    :param nodes: number of nodes in the network
    :param edges: number of edges in the network
    :param population: number of individuals in the population
    :param generation: number of generations
    :param r: crossover rate
    :return:
    """

    pop = toolbox.population()
    convergence = []
    for ind in pop:
        ind.subset = toolbox.subsets(chrom=ind)
        ind.fitness.values = (toolbox.evaluate(communities=ind.subset),)

    pop.sort(key=lambda x: x.fitness, reverse=True)    # do the evolution
    for g in range(generation):
        elites = pop[:int(population * .1)]
        size = int(np.floor(population * 0.9))
        # pop = pop[:size]
        offspring = list(map(toolbox.clone, pop[:]))
        # convergence.append(offspring[0].fitness.values[0])
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
            ind.fitness.values = (toolbox.evaluate(communities=ind.subset), )
        # map invalid_ind to offspring
        offspring = toolbox.select(offspring, k=size)
        offspring += elites
        # The population is entirely replaced by the offspring
        offspring.sort(key=lambda x: x.fitness, reverse=True)
        pop[:] = offspring[:population]
        # pop.sort(key=lambda x: x.fitness, reverse=True)
        try:
            for i, value in enumerate(convergence[g]):
                convergence[g][i] += pop[0].fitness.values[i]
        except IndexError:
            convergence.append([x for x in pop[0].fitness.values])

    best_ind = tools.selBest(pop, 1, fit_attr="fitness")[0]

    # return best_ind.fitness.values
    # return be
    print("Best individual is %s, %s" % (best_ind.subset, best_ind.fitness.values))

    return best_ind, convergence

def generate_chrom(nodes, Adj):
    chrom = np.array([], dtype=int)
    for x in nodes:
        rand = np.random.choice(nodes)
        while Adj[x, rand] < 1:
            rand = np.random.choice(nodes)
        chrom = np.append(chrom, rand)
    return creator.Individual(chrom)

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


def find_subsets(G, chrom):
    """
    Finds all subsets of a given chromsome

    :param chrom:
    :return:
    """
    temp = list(chrom[:])
    sub = []
    for x in range(len(chrom)):
        neighbours = set(G.neighbors(x))
        intersect = list(neighbours.intersection(temp))
        if len(intersect) == 0 or (len(intersect) == 1 and {x, intersect[0]} in sub):
            sub.append({x,x})
        else:
            node = np.random.choice(intersect)
            sub.append({x, node})
            temp.remove(node)

    result = sub
    candidate = merge_subsets(result)
    while candidate != result:
        result = candidate
        candidate = merge_subsets(result)

    return result

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
            mutant =  np.random.randint(0, len(chrom))
            row = Adj[mutant].toarray()[0]
            neighbor = [i for i in range(len(row)) if row[i] > 0]

            if len(neighbor) > 1:
                neighbor.remove(chrom[mutant])
                to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
                chrom[mutant] = neighbor[to_change]
                neighbor.append(chrom[mutant])
    return chrom

def evaluate(G, communities, weight='weight', resolution=1.0):
    """
    :param communities:
    :return:
    """
    return nx_comm.modularity(G, communities, weight=weight, resolution=resolution)

def register_toolbox(graph, CXPB=0.4, MUTPB=0.6, population=100):
    global toolbox
    nodes = graph.nodes()
    Adj = nx.adjacency_matrix(graph)
    # create the population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list)

    # create the toolbox
    toolbox = base.Toolbox()

    toolbox.register("individual", generate_chrom, nodes=nodes, Adj=Adj)
    toolbox.register("subsets", find_subsets, G=graph)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population)
    toolbox.register("evaluate", evaluate, G=graph, weight='weight', resolution=1.0)

    toolbox.register("mate", tools.cxUniform, indpb=CXPB)
    toolbox.register("mutate", mutation, Adj=Adj, mutation_rate=MUTPB)
    toolbox.register("select", tools.selRoulette, fit_attr="fitness")


    toolbox.register("run", community_detection)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list, centres=list)

    return toolbox

CXPB, MUTPB = 0.4, 0.6





# check degree and see if can be moved