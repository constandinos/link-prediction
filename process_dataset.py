import pandas as pd
from tqdm import tqdm

# label columns
mainFields = ['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments']
relatedVideos = ['relatedID_1', 'relatedID_2', 'relatedID_3', 'relatedID_4', 'relatedID_5', 'relatedID_6',
                 'relatedID_7', 'relatedID_8', 'relatedID_9', 'relatedID_10', 'relatedID_11', 'relatedID_12',
                 'relatedID_13', 'relatedID_14', 'relatedID_15', 'relatedID_16', 'relatedID_17', 'relatedID_18',
                 'relatedID_19', 'relatedID_20']


def read_data():
    """
    This function will read the dataset from files into a dataframe.
    Also, this function will drop rows with empty values.

    Returns
    -------
    df: numpy array
        The dataset
    """

    # read dataset into dataframes
    df_0 = pd.read_csv('dataset/0.txt', sep='\t', header=None)
    df_1 = pd.read_csv('dataset/1.txt', sep='\t', header=None)
    df_2 = pd.read_csv('dataset/2.txt', sep='\t', header=None)
    df_3 = pd.read_csv('dataset/3.txt', sep='\t', header=None)

    # concat dataframes into a single dataframe
    df = pd.concat([df_0, df_1, df_2, df_3])

    # add a label in each column
    df.columns = mainFields + relatedVideos

    # drop rows with empty values
    df = df.dropna(subset=['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments'])

    # reset index
    df.reset_index(inplace=True)
    df = df.drop(columns=['index'])

    return df


def rename_node(df):
    """
    This function will give a unique integer number to each node id.

    Parameters
    ----------
    df: numpy array
        The dataset

    Returns
    -------
    nodes_id: list
        A list with nodes id
    renaming: dict
        The corresponding between node id and a integer number
    """

    # create a list with unique nodes id
    nodes_id = list(set(df['id'].values))

    # create an empty Dictionary
    renaming = {}

    # give a unique integer number to each node
    for i in range(0, len(nodes_id)):
        renaming[nodes_id[i]] = i

    return nodes_id, renaming


def create_nodes(df, nodes_id, renaming):
    """
    This function will give a unique integer number to each node id.

    Parameters
    ----------
    df: numpy array
        The dataset
    nodes_id: list
        A list with nodes id
    renaming: dict
        The corresponding between node id and a integer number
    """

    # copy some columns to another dataframe
    nodes_df = df[['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments']].copy()

    # replace node id with integer number
    for i in tqdm(range(0, len(nodes_df['id']))):
        nodes_df.loc[nodes_df['id'][i]] = renaming[nodes_df['id'][i]]

    # write nodes in a file
    nodes_df.to_csv('nodes.csv')


def create_edges(df, nodes_id, renaming):
    """
    This function will give a unique integer number to each node id.

    Parameters
    ----------
    df: numpy array
        The dataset
    nodes_id: list
        A list with nodes id
    renaming: dict
        The corresponding between node id and a integer number
    """

    # capture edges in 2 separate lists
    node_list_1 = []
    node_list_2 = []
    for index, row in tqdm(df.iterrows()):
        n1_id = row['id']
        for related in relatedVideos:
            n2_id = row[related]
            if n2_id in nodes_id:
                node_list_1.append(renaming[n1_id])
                node_list_2.append(renaming[n2_id])

    # insert edges in a dataframe
    edges_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    # write edges a file
    edges_df.to_csv('edges.csv')

# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# read dataset
df = read_data()

# rename nodes id
nodes_id, renaming = rename_node(df)

# create nodes
create_nodes(df, nodes_id, renaming)

# create edges
create_edges(df, nodes_id, renaming)

