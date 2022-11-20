import pickle
import random
import sys

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from deap import base, creator, tools


def community_detection(pop, centrality, Adj, graph, generation=30, population=500):
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
    # pop, centrality = toolbox.initialise(graph, population)
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
        offspring = list(map(toolbox.clone, pop[:]))
        # convergence.append(offspring[0].fitness.values[0])
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1=child1, ind2=child2)
            # print("child1:", child1.centres)
            # print("child2:", child2.centres)
            toolbox.mutate(chrom=child1, G=graph, Adj=Adj, centrality=centrality)
            toolbox.mutate(chrom=child2, G=graph, Adj=Adj, centrality=centrality)
            #
            # print("child1:", child1.centres)
            # print("child2:", child2.centres)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values
            # cross two individuals with probability CXPB
            # for mutant in offspring:

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            # print("ind:", ind, ind.centres, ind.subset)
            ind.subset = toolbox.subsets(chrom=ind, G=graph, centrality=centrality)
            ind.fitness.values = toolbox.evaluate(communities=ind.subset, G=graph, centres = ind.centres)
        # map invalid_ind to offspring
        offspring += invalid_ind
        # offspring += elites
        # The population is entirely replaced by the offspring
        pop[:] = toolbox.select(offspring, k = size) + elites
        pop.sort(key=lambda x: x.fitness, reverse=True)
        try:
            for i, value in enumerate(convergence[g]):
                convergence[g][i] += pop[0].fitness.values[i]
        except IndexError:
            convergence.append([x for x in pop[0].fitness.values])

    best_ind = tools.selBest(pop, 1, fit_attr="fitness")[0]

    # return best_ind
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
    k = np.random.randint(1, len(chrom) / 3)
    # get the k nodes within the chrom as centres
    centres = list(set(np.random.choice(chrom, size=k, replace=False)))
    # print("centres:", centres)
    return centres


def generate_chrom(nodes, Adj):
    chrom = np.array(nodes, dtype=int)
    # shuffle the nodes
    np.random.shuffle(chrom)
    chrom = creator.Individual(chrom)
    chrom.centres = init_centres(chrom)
    chrom.visited = set()
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
    sub = [{c} for c in chrom.centres]
    # for i in range(len(temp)):
    #     sub.append({random.choice(chrom.centres), i})
    while temp:
        old_sub = sub[:]
        for i, c in enumerate(sub):
            neighbours = set()
            for x in c:
                neighbours = neighbours | set(G.neighbors(x))

            intersect = neighbours.intersection(temp)
            # remove values in centres from intersect
            # intersect = list(filter(lambda x: any(map(lambda y: x not in y, sub)), intersect))
            if not intersect:
                sub[i] = sub[i] | c
            else:
                sub[i] = sub[i] | intersect
                # remove intersect from temp
                temp = list(set(temp) - intersect)
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
    # # add remaining nodes to sub
    # i = 0
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
            intersect = set(subset).intersection(centres)
            if intersect:
                # c = intersect[0]
                # c = random.choice(centres)
                # get the value in centres that corresponds to the node in subset
                # get all subsets except the current one

                centrality = np.mean([nx.closeness_centrality(G.subgraph(subset), c) for c in intersect])
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
    # centres_flat = {item for sublist in centres for item in sublist}
    matrix = nx.to_numpy_matrix(G, weight=weight)
    CS = 0
    for s in communities:
        submatrix = np.zeros((len(G.nodes), len(G.nodes)), dtype=int)
        for i in s:
            for j in s:
                submatrix[i][j] = matrix[i][j]
        M = 0
        v = 0
        for row in list(s):
            row_mean = np.sum(submatrix[row]) / len(s)
            v += np.sum(submatrix[row])
            M += (row_mean ** r) / len(s)
        CS += M * v
    return CS
    modularity_score = nx_comm.modularity(G, communities, weight=weight, resolution=resolution)
    # centrality_score = calc_ind_centrality(G, communities, centres=centres, weight=weight)
    return np.mean([modularity_score,]),


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
            neighbors = list(G.neighbors(centre))
            # remove the nodes that are already in the centre
            neighbors = list(filter(lambda x: x not in chrom.centres + list(chrom.visited), neighbors))
            # find the first node in neighbours that has a higher centrality value than the current node
            # for n in neighbors:
            #     if dict(centrality)[n] > dict(centrality)[centre]:
            #         chrom.centres[c] = chrom.centres[c] - {x} | {n}
            #         break

            if not neighbors:
                # if there are no neighbours, choose a random node from the centres and combine them
                # print("rand:", rand)
                temp = centre
                while temp in chrom.centres + list(chrom.visited):
                    temp = random.choice(chrom)

                chrom.centres[c] = temp
            else:
                # neighbors.sort(key=lambda x: dict(centrality)[x], reverse=True)
                # get neighbour with max centrality that is not already in the centres
                n = max(neighbors, key=lambda x: dict(centrality)[x])
                if dict(centrality)[n] > dict(centrality)[centre]:
                    chrom.visited.add(n)
                    chrom.centres[c] = n
                else:
                    # if the neighbour is not more central than the current node, choose a random node from the centres and combine them
                    temp = centre
                    while temp in chrom.centres + list(chrom.visited):
                        temp = random.choice(chrom)
                    chrom.centres[c] = temp
                # x = min(centre, key=lambda x: dict(centrality)[x])
                # x_orig = x
                # print("x:", x)
                # print(len(neighbors))
                # for x in chrom.centres[c]:
                #     if dict(centrality)[n] > dict(centrality)[x]:
                #         centre.remove(x)
                # if len(centre) != len(chrom.centres[c]):
                #     # if the centre has not changed, choose a random node from the centres and combine them
                #     # print("rand:", rand)
                #     centre.add(n)
                # chrom.centres[c] = centre
    # REMOVE DUPLICATE CENTRES
    chrom.centres = list(set(chrom.centres))
                    # find the node in neighbours that has the highest centrality value
                    # find the node in centre that has the lowest centrality value


                        # if we have reached the last node in neighbours and the node in centre still has the highest
                        # centrality value, then we need to remove the node from the centres
                # neighbor.append(chrom[i])
    return chrom

def cxUniform(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1.centres), len(ind2.centres))
    for i in range(size):
        if random.random() < indpb:
            ind1.centres[i], ind2.centres[i] = ind2.centres[i], ind1.centres[i]

    return ind1, ind2

############################################################################################


def create(pop_size, G, Adj, CXPB=0.4, MUTPB=0.6, ):
    global toolbox
    # create the population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, subset=list, centres=list, visited=set)

    # create the toolbox
    toolbox = base.Toolbox()

    toolbox.register("subsets", find_subsets)
    toolbox.register("evaluate", evaluate, weight='weight', resolution=1.0)

    toolbox.register("mate", cxUniform, indpb=CXPB)
    toolbox.register("mutate", mutation, mutation_rate=MUTPB)
    toolbox.register("select", tools.selRoulette, fit_attr="fitness",)

    toolbox.register("individual", generate_chrom, Adj=Adj, nodes=list(G.nodes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

    toolbox.register("centrality", calculate_centrality, G=G)
    toolbox.register("run", community_detection)
    # pickle.dump(toolbox, open("toolbox.p", "wb"))
    return toolbox


if __name__ == '__main__':
    create(100, CXPB=0.4, MUTPB=0.6)
