import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graph():
    # read dataset
    df_0 = pd.read_csv('dataset/0.txt', sep='\t', header=None)
    df_1 = pd.read_csv('dataset/1.txt', sep='\t', header=None)
    df_2 = pd.read_csv('dataset/2.txt', sep='\t', header=None)
    df_3 = pd.read_csv('dataset/3.txt', sep='\t', header=None)
    df = pd.concat([df_0, df_1, df_2, df_3])
    mainFields = ['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments']
    relatedVideos = ['relatedID_1', 'relatedID_2', 'relatedID_3', 'relatedID_4', 'relatedID_5', 'relatedID_6',
                     'relatedID_7', 'relatedID_8', 'relatedID_9', 'relatedID_10', 'relatedID_11', 'relatedID_12',
                     'relatedID_13', 'relatedID_14', 'relatedID_15', 'relatedID_16', 'relatedID_17', 'relatedID_18',
                     'relatedID_19', 'relatedID_20']
    new = mainFields + relatedVideos
    df.columns = mainFields + relatedVideos

    # drop rows with empty values
    df = df.dropna(subset=['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments'])

    # add nodes to graph
    G = nx.Graph()
    nodes = df['id'].values
    G.add_nodes_from(nodes)

    # add edges to graph
    for index, row in df.iterrows():
        n1 = row['id']
        for related in relatedVideos:
            n2 = row[related]
            if G.has_node(n2):
                G.add_edge(n1, n2)

    print('number of nodes = ' + str(G.number_of_nodes()))
    print('number of edges = ' + str(G.number_of_edges()))
    nx.draw(G)
    plt.show()


load_graph()
