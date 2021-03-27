# python featurization.py petster-hamster-household_edges.csv

import networkx as nx
import pandas as pd
import sys


def jaccards_coefficient(G, df_edges, df_features):
    """
    Calculating jaccard's coefficient of nodes.

    Parameters
    ----------
    G: networkx
        The graph
    df_edges: numpy array
        All edges
    df_features: numpy array
        The edges with features
    """

    grade = []
    for source_node, destination_node in df_edges.itertuples(index=False):
        preds = nx.jaccard_coefficient(G, [(source_node, destination_node)])
        for u, v, p in preds:
            grade.append(p)
    df_features['jaccard_coef'] = grade


def adamic_adar(G, df_edges, df_features):
    """
    Calculating adamic adar coefficient of nodes.

    Parameters
    ----------
    G: networkx
        The graph
    df_edges: numpy array
        All edges
    df_features: numpy array
        The edges with features
    """

    grade = []
    for source_node, destination_node in df_edges.itertuples(index=False):
        preds = nx.adamic_adar_index(G, [(source_node, destination_node)])
        for u, v, p in preds:
            grade.append(p)
    df_features['adamic_adar'] = grade


def preferential_attachment(G, df_edges, df_features):
    """
    Calculating preferential attachment coefficient of nodes.

    Parameters
    ----------
    G: networkx
        The graph
    df_edges: numpy array
        All edges
    df_features: numpy array
        The edges with features
    """

    grade = []
    for source_node, destination_node in df_edges.itertuples(index=False):
        preds = nx.preferential_attachment(G, [(source_node, destination_node)])
        for u, v, p in preds:
            grade.append(p)
    df_features['preferential_attachment'] = grade


def clustering_coef(G, df_edges, df_features):
    """
    Calculating clustering coefficient of nodes.
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering

    Parameters
    ----------
    G: networkx
        The graph
    df_edges: numpy array
        All edges
    df_features: numpy array
        The edges with features
    """

    grade = []
    for source_node, destination_node in df_edges.itertuples(index=False):
        grade.append((nx.clustering(G, source_node)) * (nx.clustering(G, destination_node)))
    df_features['clustering_coef'] = grade


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define input and output files
filein = sys.argv[1]
fileout = filein.split('.')[0]

# read input data
df_features = pd.read_csv('../dataset/' + filein)
print('Edges with classes:')
print(df_features)
print()
df_edges = df_features[['source_node', 'destination_node']].copy()

# get possitive edges
df_pos = df_features.loc[df_features['class'] == 1]
print('Possitive edges:')
print(df_pos)
print()

# create graph
G = nx.from_pandas_edgelist(df_pos, 'source_node', 'destination_node', create_using=nx.Graph())

# get features
jaccards_coefficient(G, df_edges, df_features)
adamic_adar(G, df_edges, df_features)
preferential_attachment(G, df_edges, df_features)
clustering_coef(G, df_edges, df_features)

# reindex features
df_features = df_features.reindex(['source_node', 'destination_node', 'jaccard_coef', 'adamic_adar',
                                   'preferential_attachment', 'clustering_coef', 'class'], axis='columns')
print('Features:')
print(df_features)
print()

# write features in a file
df_features.to_csv('../dataset/' + fileout + '_features.csv', index=False)
