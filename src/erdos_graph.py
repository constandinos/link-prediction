import pandas as pd
from networkx import nx

n = 10000  # 10 nodes
m = n * 3  # 20 edges

G = nx.gnm_random_graph(n, m)

source_node_list = []
destination_node_list = []
for edge in G.edges():
    source_node_list.append(edge[0])
    destination_node_list.append(edge[1])

# create a dataframe with edges
edges_df = pd.DataFrame({'source_node': source_node_list, 'destination_node': destination_node_list})

# write edges a file
edges_df.to_csv('../dataset/erdos.txt', index=False, header=False)
