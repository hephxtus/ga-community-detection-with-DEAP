import pickle
import random
import sys

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from deap import base, creator, tools


def community_detection(pop, Adj, centrality, graph, generation=30, population=500):
    """
    :param Adj:
    :param centrality:
    :param pop:
    :param graph:
    :param population: number of individuals in the population
    :param generation: number of generations
    :param r: crossover rate
    :return:
    """
    convergence = []
    for ind in pop:
        ind.subset = toolbox.subsets(chrom=ind, G=graph, centrality=centrality)
        ind.fitness.values = toolbox.evaluate(communities=ind.subset, G=graph, centres=ind.centres)

    pop.sort(key=lambda x: x.fitness, reverse=True)
    # do the evolution
    for g in range(generation):
        elites = pop[:int(population * .1)]
        size = int(np.floor(population * 0.9))
        # pop = pop[:size]
        offspring = toolbox.select(pop, size)
        offspring = list(map(toolbox.clone, offspring))
        # convergence.append(offspring[0].fitness.values[0])
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1=child1.centres, ind2=child2.centres)
            toolbox.mutate(chrom=child1, G=graph, Adj=Adj, centrality=centrality)
            toolbox.mutate(chrom=child2, G=graph, Adj=Adj, centrality=centrality)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values
            # cross two individuals with probability CXPB
            # for mutant in offspring:

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.subset = toolbox.subsets(chrom=ind, G=graph, centrality=centrality)
            ind.fitness.values = toolbox.evaluate(communities=ind.subset, G=graph, centres = ind.centres)
        # map invalid_ind to offspring
        offspring += invalid_ind
        offspring += elites
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        pop.sort(key=lambda x: x.fitness, reverse=True)
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


def calculate_centrality(G):
    """
    use
    :param G: networkx graph
    :return: (node, centrality) pairings
    """
    centrality = nx.eigenvector_centrality(G)
    return centrality


def init_centres(chrom):
    """
    Finds the node with the highest degree in the given chromsome

    :param chrom:
    :return:
    """
    # generate random number (k) of centres
    k = np.random.randint(1, len(chrom) / 2)
    # get the k nodes within the chrom as centres
    centres = list(set(np.random.choice(chrom, size=k, replace=False)))
    centres = [{c} for c in centres]
    return centres


def generate_chrom(nodes, Adj):
    chrom = np.array(nodes, dtype=int)
    chrom = creator.Individual(chrom)
    chrom.centres = init_centres(chrom)
    return chrom


def find_central_node(G, chrom):
    """
    Finds the node with the highest degree in the given chromsome

    :param chrom:
    :return:
    """
    degrees = G.degree(chrom)
    return np.max(degrees, key=lambda x: x[1])[0]


def merge_subsets(sub):
    # sys.exit()
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
    # for x in range(len(chrom)):
    #     neighbours = set(G.neighbors(x))
    #     intersect = list(neighbours.intersection(temp))
    #     if len(intersect) == 0 or (len(intersect) == 1 and {x, intersect[0]} in sub):
    #         sub.append({x, x})
    #     else:
    #         # print list of nodes in node_centralities
    #         # if node centrality does not intersect with intersect, print intersect
    #         if not any(map(lambda n: n[0] in intersect, node_centralities)):
    #             print("INTERSECT:", intersect)
    #         node = max(list(filter(lambda n: n[0] in intersect, node_centralities)), key=lambda n: n[1])[0]
    #
    #         sub.append({x, node})
    #         # remove node from temp
    #         temp.remove(node)
    #         # remove node from node_centralities
    #         node_centralities = list(filter(lambda n: n[0] != node, node_centralities)) if node not in temp \
    #             else node_centralities
    #
    #
    sub = chrom.centres
    while temp:
        old_sub = sub
        for i, c in enumerate(sub):
            neighbours = set()
            for x in c:
                neighbours = neighbours | set(G.neighbors(x))

            intersect = list(neighbours.intersection(temp))
            # remove values in centres from intersect
            intersect = list(filter(lambda x: any(map(lambda y: x not in y, sub)), intersect))
            if len(intersect) == 0:
                sub[i] = sub[i] | c
            else:
                sub[i] = sub[i] | set(intersect)
                # remove intersect from temp
                temp = list(set(temp) - set(intersect))
            # remove c from temp
            if set(temp) & c:
                temp = list(set(temp) - c)
        if old_sub == sub:
            for t in temp:
                neighbours = set(G.neighbors(t))
                if len(neighbours) == 0:
                    sub.append({t, t})
                    # temp.remove(t)
                else:
                    sub.append({t, random.choice(list(neighbours))})
                # sub.append({t, t})
                temp.remove(t)
            # temp.remove(c)
    # add remaining nodes to sub
    i = 0
    # while temp:
    #     old_sub = sub
    #     for t in temp:
    #         # find all nodes immediately connected to it within temp
    #         neighbours = set(G.neighbors(t))
    #         intersect = set(neighbours.intersection(temp))
    #         # if intersect is empty
    #         if len(intersect) == 0:
    #             sub.append({t, t})
    #             temp.remove(t)
    #         # check if any neighbours are in any of the subsets
    #         for s in sub:
    #             if s & intersect:
    #                 s.add(t)
    #                 temp.remove(t)
    #                 break
    #     if old_sub == sub:

    # for x in range(len(chrom)):
    #     neighbours = set(G.neighbors(x))
    #     intersect = list(neighbours.intersection(temp))
    #     if len(intersect) == 0 or (len(intersect) == 1 and {x, intersect[0]} in sub):
    #         sub.append({x, x})
    #     else:
    #         node = random.choice(temp)
    #         sub.append({x, node})
    #         # remove node from temp
    #         temp.remove(node)
    #         # remove node from node_centralities

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
    return np.max(closeness, key=lambda x: x[1])[0]


def calc_ind_centrality(G, chrom, weight, centres):
    """
    Finds the node with the highest centrality in the given chromsome
    :param chrom:
    :return:
    """
    centralities = []
    # for c in centres:
    #     centrality = nx.eigenvector_centrality(G.subgraph(list), c, weight=weight)
    #     centralities.append((c, centrality))
    for subset in chrom:
        if len(subset) < 2:
            centralities.append(0)
        else:
            # check if any of the nodes in the subset are in the centres
            # if so, get the centre
            # else, calculate the centrality of the subset
            intersect = list(set(subset).intersection(centres))
            if intersect:
                c = intersect[0]
                # c = random.choice(centres)
                # get the value in centres that corresponds to the node in subset
                # get all subsets except the current one
                all_subsets = list(filter(lambda x: x != subset, chrom))
                # # get all nodes in all subsets
                all_nodes = set()
                for s in all_subsets:
                    all_nodes = all_nodes.union(s)

                centrality = nx.closeness_centrality(G.subgraph(subset), c) - nx.closeness_centrality(G, c)
                centralities.append(centrality)
            else:
                try:
                    centrality = nx.eigenvector_centrality(G.subgraph(list(subset)), max_iter=500, weight=weight)
                except nx.exception.PowerIterationFailedConvergence as e:
                    centralities.append(0)
                    print(e)
                    print("subset:", subset)
                    continue
                centralities.append(list(dict(np.max(centrality, axis=0)).values())[0])
    return np.mean(centralities)


def evaluate(G, communities, centres, weight="weight", resolution=1):
    """
    Calculates the modularity of the given communities

    :param communities:
    :return:
    """
    # centres_flat = [item for sublist in centres for item in sublist]
    modularity_score = nx_comm.modularity(G, communities, weight=weight, resolution=resolution)
    # centrality_score = calc_ind_centrality(G, communities, centres=centres_flat, weight=weight)
    return modularity_score,


def mutation(chrom, Adj, mutation_rate, G, centrality):
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

    # if np.random.random_sample() < mutation_rate:
    #     chrom = chrom
    #     neighbor = []
    #     while len(neighbor) < 2:
    #         # find node with highest centrality value in chrom
    #         # mutant = max(chrom, key=lambda x: G.nodes[x]['centrality'])
    #         mutant = np.random.choice(chrom)
    #         row = Adj[mutant].toarray()[0]
    #         neighbor = [i for i in range(len(row)) if row[i] > 0]
    #
    #         if len(neighbor) > 1:
    #             neighbor.remove(chrom[mutant])
    #             to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
    #             chrom[mutant] = neighbor[to_change]
    #             neighbor.append(chrom[mutant])
    # REWRITE THIS to be more efficient
    for c, centre in enumerate(chrom.centres):
        if np.random.random_sample() < mutation_rate:
            neighbors = set()
            for x in centre:
                neighbors = neighbors | set(G.neighbors(x))
            neighbors = list(neighbors - centre)

            # row = Adj[n].toarray()[0]
            # neighbor = [i for i in range(len(row)) if row[i] > 0]
            # print(chrom.subset)
            while len(neighbors) > 0:
                n = int(np.floor(np.random.random_sample() * (len(neighbors))))
                to_change = random.choice(list(centre))

                # find node in neighbor with highest centrality value
                # to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
                if dict(centrality)[neighbors[n]] > dict(centrality)[to_change]:
                    chrom.centres[c] = {neighbors[n]}
                    break
                else:
                    # remove node at index to_change from neighbor
                    neighbors.pop(n)
                if not neighbors:
                    rnad = random.choice(chrom.centres)
                    chrom.centres[c] = set(centre).union(set(rnad))
                    # remove random node from chrom.centres
                    chrom.centres.remove(rnad)
                # neighbor.append(chrom[i])
    return chrom


############################################################################################
CXPB, MUTPB = 0.4, 0.6


def initialize(G):
    global toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list, centres=list)
    Adj = nx.adjacency_matrix(G)
    centrality = calculate_centrality(G).items()
    toolbox.register("individual", generate_chrom, Adj=Adj, nodes=list(G.nodes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=100)
    pop = toolbox.population()
    return pop, Adj, centrality


def create():
    global toolbox
    # create the population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list, centres=list)

    # create the toolbox
    toolbox = base.Toolbox()

    toolbox.register("subsets", find_subsets)
    toolbox.register("evaluate", evaluate, weight='weight', resolution=1.0)

    toolbox.register("mate", tools.cxUniform, indpb=CXPB)
    toolbox.register("mutate", mutation, mutation_rate=MUTPB)
    toolbox.register("select", tools.selRoulette, fit_attr="fitness")

    toolbox.register("run", community_detection)
    toolbox.register("initialise", initialize)
    pickle.dump(toolbox, open("toolbox.p", "wb"))
    return toolbox


if __name__ == '__main__':
    create()
