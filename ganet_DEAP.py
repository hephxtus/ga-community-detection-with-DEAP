import glob
import itertools
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from os import listdir
from os.path import isfile, join

import numpy as np
import networkx as nx
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
from cdlib import algorithms

import networkx.algorithms.community as nx_comm
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

import centrality_community_detection

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


def draw_communities(communities, G, title='TEST', metrics={}):
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
    # print(list(map(lambda x: cmap[communities.index(x)], communities)))
    # nx.draw_networkx(G, node_color=list(map(lambda x: cmap[communities.index(x)], communities)), with_labels=True)
    for i, nodes in enumerate(communities):
        for node in nodes:
            node_cmap.append((node, {"color": cmap[i]}))

    modularity = nx_comm.modularity(G, communities)
    centrality = centrality_community_detection.calc_ind_centrality(G, communities, weight='weight', centres=[])
    for key in range(len(communities)):
        sub_graph = G.subgraph(communities[key])
        nx.draw_networkx(sub_graph, pos=nx.spring_layout(G), node_color=cmap[key], with_labels=True)
    plt.title(list(metrics.items()))
    # for i, (key, value) in enumerate(metrics.items()):
    #     plt.text(0.5, 0.5 - i * 0.1, f"{key}: {value:.4f}", fontsize=12)
    # plt.annotate(f'Modularity: {modularity}', xy=(0.5, 0.05), xycoords='figure fraction',
    #              horizontalalignment='center', )
    # plt.annotate(f'Centrality: {centrality}', xy=(0.5, 0.01),
    #                 xycoords='figure fraction', horizontalalignment='center', )
    plt.savefig(f"output/{title}")


def classify_communities(true, pred, nodes_list: list):

    classified = []
    for i, community in enumerate(pred):
        for node in community:
            classified.append(i)
    #     for i, labels in enumerate(true):
    #         # check if any of the nodes in the label is in the community
    #         if any(node in community for node in labels) and i not in classified.keys():
    #             classified[i] = community
    #             break
    #         elif i == len(true) - 1:
    #             classified[len(classified)] = community
    #
    # for node in range(len(nodes_list)):
    #     # find node in classified
    #     for i, community in classified.items():
    #         if nodes_list[node] in list(community):
    #             nodes_list[node] = i
    #             break
    return classified


def evaluate_run(G, pred, start_time = time.time(), true=None):
    pred_labels = classify_communities(true=true, pred=pred, nodes_list=list(G.nodes))
    metrics = {}
    if true:
        true_labels = classify_communities(true=true, pred=true, nodes_list=list(G.nodes))

        metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
        metrics['Accuracy'] = accuracy_score(true_labels, pred_labels)
        metrics['Community Diff'] = abs(len(true) - len(pred))
    else:

        metrics['NMI'] = 0
        metrics['Accuracy'] = 0
        metrics['Community Diff'] = 0

    metrics['Modularity'] = nx_comm.modularity(G, pred)
    metrics['Centrality'] = centrality_community_detection.calc_ind_centrality(G, pred, weight='weight', centres=[])

    metrics['Time'] = time.time() - start_time

    return metrics


def calc_metric(metric: list) -> dict:
    return {
        "mean": round(np.mean(metric), 4),
        "std": round(np.std(metric), 4),
        "min": round(np.min(metric), 4),
        "max": round(np.max(metric), 4),
        "median": round(np.median(metric), 4),
    }


def evaluate_fits(fits):
    # get all NMI scores
    metric_scores = {}
    for metric in fits[0].keys():
        scores = [fit[metric] for fit in fits]
        metric_scores[metric] = calc_metric(scores)
        # metric_scores.append([fit[metric] for fit in fits])

    return pd.DataFrame(metric_scores, index=["mean", "std", "min", "max", "median"], columns=metric_scores.keys())


# def run_community_detection(G, algorithm):
#     if algorithm == "louvain":
#     return algorithm(G)
#     pass
#
#
# def run_experiment(G, true, true_labels, max_iterations=10, algorithm=toolbox.run):
#     fits = []
#     for i in range(max_iterations):
#         print(f"Run {i}")
#         pred = list(algorithm(G))
#         metrics = evaluate_run(G, true, true_labels, pred)
#         fits.append(metrics)
#     return fits


# set random seed
np.random.seed(389)

# graph = nx.karate_club_graph()
# {filename:graph} for all ffiles in the databaase folder, gml vs txt
# graphs = {f: nx.read_gml(f) for f in glob.glob("database/*.gml")}
graphs = {os.path.splitext(f)[0]: nx.read_gml(os.path.join("database", f), label='id') for f in os.listdir("database")
          if f.endswith(".gml")}

print("Graphs:", graphs.keys())
# read gml of all files

# exit()

dolphins = graphs["dolphins"]
zachary_karate = nx.karate_club_graph()
# dolphins = nx.convert_node_labels_to_integers(dolphins, first_label=0, ordering='default', label_attribute=None)
network_scientists = graphs["netscience"]
# network_scientists = nx.convert_node_labels_to_integers(network_scientists, first_label=0, ordering='default', label_attribute=None)
facebook = nx.read_edgelist('database/facebook_combined.txt', create_using=nx.Graph(), nodetype=int)
facebook = nx.convert_node_labels_to_integers(facebook, first_label=0, ordering='default', label_attribute=None)
# use dolphins for testing
graph = dolphins
# pos = nx.spring_layout(dolphins)
# pickle load toolbox if it exists else create it
# if not os.path.exists('toolbox.p'):
toolbox = centrality_community_detection.create()
# else:
#     toolbox = pickle.load(open('toolbox.p', 'rb'))
# toolbox = pickle.load(open("toolbox.p", "rb"))

# graph = dolphins
# nx.draw_networkx(graph, pos, node_size=75, alpha=0.8)
# plt.show()
nodes = graph.nodes
edges = graph.edges
i = 0
max_iterations = 5
true_communities = None
# true_labels = set(nx.get_node_attributes(graph, 'club').values())
# # get index of first node in each club
# true_communities = [set([node for node, data in graph.nodes(data=True) if data['club'] == club]) for club in
#                     true_labels]
# false_communities = [set([node for node, data in graph.nodes(data=True) if data['club'] != club]) for club in
#                         true_labels]
# print("True Communities:", true_communities)
# print("False Communities:", false_communities)
# true_labels = classify_communities(true_communities, false_communities, list(graph.nodes))
# # get metrics for true communities
# true_metrics = evaluate_run(graph, pred=false_communities, true = true_communities)
# print(true_metrics)
# draw_communities(communities=true_communities, G=graph, metrics={}, title="ACTUAL")
fits = []
times = []


# true_label_index = {label: i for i, label in enumerate(true_labels)}
while i < max_iterations:
    start = time.time()
    communities_lp = list(nx_comm.label_propagation_communities(graph))
    fits.append(evaluate_run(G=graph, pred=communities_lp, true=true_communities, start_time=start))
    times.append(time.time() - start)
    i += 1
print("LABEL PROPOGATION COMMUNITIES:", communities_lp)
metrics = evaluate_fits(fits)

print("EVALUATION METRICS\n", evaluate_fits(fits))
draw_communities(communities=communities_lp, G=graph, metrics={"NMI": metrics["NMI"]["mean"]}, title="LABEL PROPAGATION")

fits = []
times = []
i = 0

while i < max_iterations:
    start = time.time()
    communities_louvain = nx_comm.louvain_communities(graph)
    fits.append(evaluate_run(G=graph, pred=communities_louvain, true=true_communities, start_time=start))
    times.append(time.time() - start)
    i += 1
print("LOUVAIN COMMUNITIES:", communities_louvain)
metrics = evaluate_fits(fits)

print("EVALUATION METRICS\n", evaluate_fits(fits))
draw_communities(communities=communities_louvain, G=graph, metrics={"NMI": metrics["NMI"]["mean"]}, title="LOUVAIN")

fits = []
times = []
i = 0
while i < max_iterations:
    start = time.time()
    communities_greedy = nx_comm.greedy_modularity_communities(graph)
    fits.append(evaluate_run(G=graph, pred=communities_greedy, true=true_communities, start_time=start))
    times.append(time.time() - start)
    i += 1
print("GREEDY COMMUNITIES:", communities_greedy)
metrics = evaluate_fits(fits)

print("EVALUATION METRICS\n", evaluate_fits(fits))
draw_communities(communities=communities_greedy, G=graph, metrics={"NMI": metrics["NMI"]["mean"]}, title="GREEDY")

fits = []
times = []
i = 0
while i < max_iterations:
    """implement leiden community detection"""
    start = time.time()
    communities_leiden = algorithms.leiden(graph)
    fits.append(evaluate_run(G=graph, pred=communities_leiden.communities, true=true_communities, start_time=start))
    times.append(time.time() - start)
    i += 1
print("LEIDEN COMMUNITIES", communities_leiden.communities)
metrics = evaluate_fits(fits)

print("EVALUATION METRICS\n", evaluate_fits(fits))
draw_communities(communities=communities_leiden.communities, G=graph, metrics={"NMI": metrics["NMI"]["mean"]}, title="LEIDEN")
best = (true_communities, 0)
if not true_communities:
# get the best community detection algorithm
    for communities in [communities_lp, communities_louvain, communities_greedy, communities_leiden.communities]:
        # get metrics for true communities
        true_metrics = evaluate_run(graph, communities)
        if true_metrics["Modularity"] > best[1]:
            best = (communities, true_metrics["Modularity"])
    # print(true_metrics)
    # draw_communities(communities=communities, G=graph, metrics={}, title="ACTUAL")
fits = []
times = []
i = 0

fig = plt.figure()
plt.xlabel('Generation')
plt.ylabel('Modularity')
plt.title('Modularity Convergence')

generation = 100
population = 100
convergence = [[0, 0]] * generation
# zachary_karate_true_labels = [{1, 2, 3, 4, 5, 6, 7, 8, }]

# total_time = 0
try:
    for i in range(1, max_iterations + 1):
        start = time.time()
        # interval = time.time()
        pop, Adj, centrality = toolbox.initialise(graph)
        scores, conv_score = toolbox.run(pop=pop, Adj=Adj, centrality=centrality, graph=graph, generation=generation,
                                         population=population)
        convergence = [[convergence[g][index] + conv_score[g][index] for index in range(len(conv_score[g]))] for g in
                       range(len(conv_score))]
        print(f"Iteration {i} of {max_iterations} completed")
        fits.append(evaluate_run(G=graph, pred=scores.subset, true=best[0], start_time=start))
        print(fits[-1])
        print(f"Time taken: {time.time() - start}")
finally:
    print("GA COMMUNITIES", scores.subset)
    print(evaluate_fits(fits))
    for c, _ in enumerate(convergence):
        for index in range(len(convergence[c])):
            convergence[c][index] /= i
    print("final convergence=", convergence)
    # plot all convergence index 0
    for index in range(len(convergence[0])):
        plt.plot([c[index] for c in convergence], label=f"Convergence {index}")
        plt.show()


    metrics = {"NMI": evaluate_fits(fits)['NMI']['mean']}
    draw_communities(communities=scores.subset, G=graph, metrics=metrics, title="GADeap")

# check degree and see if can be moved
