import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def community_detection(nodes, edges, population=300, generation=30, r=1.5):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    Adj = nx.adjacency_matrix(graph)
    nodes_length = len(graph.nodes())

    d = {"chrom": [generate_chrom(nodes_length, Adj) for n in range(population)]}
    dframe = pd.DataFrame(data=d)
    # return
    dframe["subsets"] = dframe["chrom"].apply(find_subsets)
    dframe["community_score"] = dframe.apply(lambda x: community_score(x["chrom"], x["subsets"], r, Adj), axis=1)
    gen = 0
    population_count = population
    while gen < generation:
        for i in range(int(np.floor(population / 10))):
            p1 = 0
            p2 = 0
            elites = dframe.sort_values("community_score", ascending=True)[int(np.floor(population / 10)):]
            # print(len(elites))
            p1 = roulette_selection(elites)
            p2 = roulette_selection(elites)
            child = uniform_crossover(dframe["chrom"][p1], dframe["chrom"][p2], 0.8)
            # print(child)
            child = mutation(child, Adj, 0.2)
            child_subsets = find_subsets(child)
            child_cs = community_score(child, child_subsets, r, Adj)
            dframe.loc[population_count] = [child, child_subsets, child_cs]
            population_count += 1
        dfsorted = dframe.sort_values("community_score", ascending=False)
        to_drop = dfsorted.index[population:]
        dframe.drop(to_drop, inplace=True)
        gen += 1
    sorted_df = dframe.sort_values("community_score", ascending=False).index[0]

    nodes_subsets = dframe["subsets"][sorted_df]
    nodes_list = list(graph.nodes())
    result = []
    for subs in nodes_subsets:
        subset = []
        for n in subs:
            subset.append(nodes_list[n])
        result.append(subset)
    return result, community_score(dframe["chrom"][sorted_df], nodes_subsets, r, Adj)


def generate_chrom(nodes_length, Adj):
    chrom = np.array([], dtype=int)
    for x in range(nodes_length):
        rand = np.random.randint(0, nodes_length)
        while Adj[x, rand] != 1:
            rand = np.random.randint(0, nodes_length)

        chrom = np.append(chrom, rand)
    return chrom


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


def find_subsets(chrom):
    """
    Finds all subsets of a given chromsome

    :param chrom:
    :return:
    """
    # print(chrom)
    sub = [{x, chrom[x]} for x in range(len(chrom))]
    # print(sub)
    result = sub
    i = 0
    while i < len(result):
        candidate = merge_subsets(result)
        if candidate != result:
            result = candidate
        else:
            break
        result = candidate
        i += 1
    # print(result)
    return result


def community_score(chrom, subsets, r, Adj):
    """
    Calculates the community score of a given chromsome
    :param chrom:
    :param subsets:
    :param r:
    :param Adj:
    :return:

    """
    matrix = Adj.toarray()
    CS = 0
    for s in subsets:
        submatrix = np.zeros((len(chrom), len(chrom)), dtype=int)
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
    # Q = 0
    # nx.set_edge_atAtributes(G, {e: 1 for e in G.edges}, 'weight')
    # A = nx.to_scipy_sparse_matrix(G).astype(float)
    # # for undirected graphs, in and out treated as the same thing
    # out_degree = in_degree = dict(nx.degree(G))
    # M = 2. * (G.number_of_edges())
    # print("Calculating modularity for undirected graph")
    #
    # nodes = list(G)
    # Q = np.sum([A[i, j] - in_degree[nodes[i]] * \
    #             out_degree[nodes[j]] / M \
    #             for i, j in product(range(len(nodes)), range(len(nodes)))
    #             if subsets[nodes[i]] == subsets[nodes[j]]])
    # return Q / M


def roulette_selection(df_elites):
    prob = np.random.random_sample()
    sum_cs = np.sum(df_elites["community_score"])
    x = 0
    selected = 0
    for i in df_elites.index:
        x += df_elites["community_score"][i]
        X = x / sum_cs
        if prob < X:
            chosen = i
            break
    return chosen


def uniform_crossover(parent_1, parent_2, crossover_rate):
    if np.random.random_sample() < crossover_rate:
        length = len(parent_1)
        mask = np.random.randint(2, size=length)
        child = np.zeros(length, dtype=int)
        for i in range(len(mask)):
            if mask[i] == 1:
                child[i] = parent_1[i]
            else:
                child[i] = parent_2[i]
        return child
    else:
        return np.array([])


def mutation(chrom, Adj, mutation_rate):

    if np.random.random_sample() < mutation_rate:
        length = len(chrom)
        mask = np.random.randint(2, size=length)
        mutated_chrom = np.zeros(length, dtype=int)
        for i in range(len(mask)):
            if mask[i] == 1:
                mutated_chrom[i] = chrom[i]
            else:
                mutated_chrom[i] = np.random.randint(0, length)
        return mutated_chrom
    else:
        return chrom

    # if np.random.random_sample() < mutation_rate:
    #     chrom = chrom
    #     neighbor = []
    #     while len(neighbor) < 2:
    #         print(chrom)
    #         mutant =  np.random.randint(1, len(chrom))
    #         # print(Adj[mutant])
    #         row = Adj[mutant].toarray()[0]
    #         # print(row)
    #         neighbor = [i for i in range(len(row)) if row[i] == 1]
    #         print("chrom:",chrom)
    #         print(mutant)
    #
    #         if len(neighbor) > 1:
    #
    #             # print("yes")
    #             print(neighbor, chrom[mutant])
    #             neighbor.remove(chrom[mutant])
    #             to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
    #             chrom[mutant] = neighbor[to_change]
    #             neighbor.append(chrom[mutant])
    #             # sys.exit()
    # return chrom

graph = nx.karate_club_graph()
pos = nx.spring_layout(graph)
nx.draw(graph, pos, node_size=75, alpha=0.8)
plt.show()
nodes = graph.nodes
edges = graph.edges


start = time.time()
for i in range(10):
    interval = time.time()
    print(community_detection(nodes, edges))
    print("Time taken: ", time.time()-interval)
end = time.time()
print("average: %d" % ((end-start)/10))

