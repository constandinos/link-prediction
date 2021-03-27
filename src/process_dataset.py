# python process_dataset.py petster-hamster-household.txt
# python process_dataset.py soc-hamsterster.txt

from sklearn.utils import shuffle
import networkx as nx
import pandas as pd
import random
import sys


def read_data(filename):
    """
    This function will read the dataset from file into a dataframe.

    Parameters
    ----------
    filename: str
        The name of dataset directory

    Returns
    -------
    dataset_df: numpy array
        The dataset
    """

    # read dataset into dataframes
    dataset_df = pd.read_csv(filename, sep='[ \t,]', engine='python', header=None)
    dataset_df = dataset_df.iloc[:, 0:2]
    dataset_df.columns = ['source_node', 'destination_node']

    # dropping duplicate values
    dataset_df = dataset_df.drop_duplicates()

    print('Dataset:')
    print(dataset_df)
    print()

    return dataset_df


def get_largest_component(dataset_df):
    """
    This function will get the largest connected component from a graph.

    Parameters
    ----------
    dataset_df: numpy array
        The dataset

    Returns
    -------
    edges_df: numpy array
        The edges of the largest connected component
    """

    # create graph
    G = nx.from_pandas_edgelist(dataset_df, 'source_node', 'destination_node', create_using=nx.Graph())

    # get the number of connected components
    num_connected_components = nx.number_connected_components(G)
    print('Graph info:')
    print(print(nx.info(G)))
    print('Number of connected components: ' + str(num_connected_components))
    print()

    # find the largest connected component
    if num_connected_components > 1:
        largest_cc = len(max(nx.connected_components(G), key=len))
        print('Size of largest connected component: ' + str(largest_cc))
        print()
        for c in nx.connected_components(G):
            if len(c) == largest_cc:
                S = G.subgraph(c).copy()
                print('Largest connected component info:')
                print(print(nx.info(S)))
                print()
                break

        # get the edges from largest connected component
        source_node_list, destination_node_list = [], []
        for edge in S.edges.data():
            source_node_list.append(edge[0])
            destination_node_list.append(edge[1])

        # create a dataframe with edges
        edges_df = pd.DataFrame({'source_node': source_node_list, 'destination_node': destination_node_list})

        print('Info about largest connected component:')
        print(print(nx.info(S)))
        print()

    else:
        edges_df = dataset_df

    print('Edges of largest connected component:')
    print(edges_df)
    print()

    return edges_df


def create_nodes(dataset_df, fileout):
    """
    This function will write in a file the unique nodes and will write them into a file.

    Parameters
    ----------
    edges_df: numpy array
        The edges of the largest connected component
    fileout: str
        The name of output file
    """

    # create a list with all unique nodes
    nodes_list = list(dataset_df['source_node'].values) + list(dataset_df['destination_node'].values)
    unique_nodes = list(set(nodes_list))

    # create a dataframe with unique nodes
    nodes_df = pd.DataFrame({'nodes': unique_nodes})

    # write unique nodes in a dataframe
    nodes_df.to_csv('../dataset/' + fileout + '_nodes.csv', index=False)


def create_edges(dataset_df, fileout):
    """
    This function will create the edges between nodes and will write them into a file.

    Parameters
    ----------
    edges_df: numpy array
        The edges of the largest connected component
    fileout: str
        The name of output file
    """

    # write edges a file
    dataset_df.to_csv('../dataset/' + fileout + '_edges.csv', index=False)


def generate_negative_edges(df_pos, G):
    """
    Generating some edges which are not present in graph for supervised learning.

    Parameters
    ----------
    df_pos: numpy array
        The possitive edges
    G: networkx
        The graph

    Returns
    -------
    df_neg: numpy array
        The negative edges
    """

    # get the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # create a dictionary with possitive edges
    edges_dict = dict()
    for edge in df_pos.values:
        edges_dict[(edge[0], edge[1])] = 1

    # create a set with a sample of random edges which are not in graph and whose shortest path is greater than 2
    missing_edges = set([])
    while len(missing_edges) < num_edges:
        a = random.randint(1, num_nodes)
        b = random.randint(1, num_nodes)
        if b < a:
            temp = a
            a = b
            b = temp
        tmp = edges_dict.get((a, b), -1)
        if tmp == -1 and a != b:
            if G.has_node(a) and G.has_node(b):
                if not nx.has_path(G, a, b) or nx.shortest_path_length(G, source=a, target=b) > 2:
                    missing_edges.add((a, b))
            else:
                continue

    # create a dataframe with negative edges
    df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])
    print('Negative edges:')
    print(df_neg)
    print()

    return df_neg


def create_all_edges(df_pos, df_neg):
    """
    Append possitive and negative edges.

    Parameters
    ----------
    df_pos: numpy array
        The possitive edges
    df_neg: numpy array
        The negative edges
    """

    # define class on edges (0 - negative edges and 1 - possitive edges)
    df_neg['class'] = 0
    df_pos['class'] = 1

    # append edges
    df_class = df_neg.append(df_pos)

    # suffle edges
    df_class = shuffle(df_class)

    # write edges a file
    df_class.to_csv('../dataset/' + fileout + '_edges.csv', index=False)
    print('All edges:')
    print(df_class)
    print()


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define input and output files
filein = sys.argv[1]
fileout = filein.split('.')[0]

# read dataset
dataset_df = read_data('../dataset/' + filein)

# find the larget connected component on graph
# df_pos = get_largest_component(dataset_df)

# create nodes
# create_nodes(df_pos, fileout)

# create edges
# create_edges(dataset_df, fileout)

# create graph
G = nx.from_pandas_edgelist(dataset_df, 'source_node', 'destination_node', create_using=nx.Graph())

print('Info about graph:')
print(print(nx.info(G)))

# generate negative edges
df_neg = generate_negative_edges(dataset_df, G)

# create all edges
create_all_edges(dataset_df, df_neg)
