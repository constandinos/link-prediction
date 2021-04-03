#######################################################################################################################
# erdos_graph.py
# This class create a random undirect graph (erdos graph).
#
# Execution commands:
# python erdos_graph.py <number of nodes> <number of edges>
# eg. python erdos_graph.py 10000 30000
#
# Created by: Constandinos Demetriou, 2021
#######################################################################################################################

import sys
import pandas as pd
from networkx import nx


def create_graph(n, m):
    """
    This function will create a random graph.

    Parameters
    ----------
    n: int
        The number of nodes
    m: int
        The number of edges

    Returns
    -------
    source_node_list: list of int
        List this source node id
    destination_node_list: list of int
        List this destination node id
    """

    G = nx.gnm_random_graph(n, m)

    source_node_list = []
    destination_node_list = []
    for edge in G.edges():
        source_node_list.append(edge[0])
        destination_node_list.append(edge[1])

    return source_node_list, destination_node_list


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define number of nodes and number of edges
n = int(sys.argv[1])
m = int(sys.argv[2])

# create the random graph
source_node_list, destination_node_list = create_graph(n, m)

# create a dataframe with edges
edges_df = pd.DataFrame({'source_node': source_node_list, 'destination_node': destination_node_list})

# write edges a file
edges_df.to_csv('../dataset/erdos.txt', index=False, header=False)
