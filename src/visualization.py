#######################################################################################################################
# visualization.py
# This class visualize the graph and plot degree distribution.
#
# Execution commands:
# python visualization.py <name of dataset with edges>
# eg. python visualization.py hamsterster_edges.csv
# <name of dataset with edges> = {hamsterster_edges.csv, twitch_edges.csv, github_edges.csv, deezer_edges.csv
#                                 facebook_edges.csv, erdos_edges.csv}
#
# Created by: Constandinos Demetriou, 2021
#######################################################################################################################

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys


def build_network(filename):
    """
    This function will read the edges from file into a dataframe and will build the network.

    Parameters
    ----------
    filename: graph
        The network

    Returns
    -------
    G: networkx graph
        The network
    """

    # read edges
    fb_df = pd.read_csv(filename)

    # create graph
    G = nx.from_pandas_edgelist(fb_df, 'source_node', 'destination_node', create_using=nx.Graph())

    return G


def draw_network(G):
    """
    This function will draw the network.

    Parameters
    ----------
    G: networkx graph
        The network
    """

    # plot the network
    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)
    plt.savefig('../results/network.png')
    plt.clf()


def degree_sequence(G):
    """
    Degree sequence of graph G

    This function returns the degree sequence of a graph G.

    Parameters
    ----------
    G: networkx graph
        The network

    Returns
    -------
    l : list
        Degree sequence
    """

    return [d for n, d in G.degree()]


def plot_degrees(degree_sequence):
    """
    Plots degree sequence.

    Parameters
    ----------
    degree_sequence: list
        Degree sequence
    """

    plt.hist(degree_sequence)
    plt.title('Degree histogram')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    plt.savefig('../results/histogram.png')
    plt.clf()
    plt.loglog(sorted(degree_sequence, reverse=True), 'b-', marker='o')
    plt.title('Degree rank plot')
    plt.xlabel('rank')
    plt.ylabel('degree')
    plt.savefig('../results/degree_rank.png')
    plt.clf()


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define input files
filein = sys.argv[1]

# build graph
G = build_network('../dataset/' + filein)

# draw network
# draw_network(G)

# plot degree distribution
deg = degree_sequence(G)
plot_degrees(deg)
