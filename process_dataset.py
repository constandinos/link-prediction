import pandas as pd
from tqdm import tqdm

# label columns

mainFields = ['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments']
relatedVideos = ['relatedID_1', 'relatedID_2', 'relatedID_3', 'relatedID_4', 'relatedID_5', 'relatedID_6',
                 'relatedID_7', 'relatedID_8', 'relatedID_9', 'relatedID_10', 'relatedID_11', 'relatedID_12',
                 'relatedID_13', 'relatedID_14', 'relatedID_15', 'relatedID_16', 'relatedID_17', 'relatedID_18',
                 'relatedID_19', 'relatedID_20']
# datasets directories
datasets = ['dataset/0.txt', 'dataset/1.txt', 'dataset/2.txt', 'dataset/3.txt']


def read_data(filenames):
    """
    This function will read the dataset from files into a dataframe.
    Also, this function will drop rows with empty values. Moreover,
    this function will write dataset in a file.

    Parameters
    ----------
    filenames: list
        A list with the datasets directories

    Returns
    -------
    df: numpy array
        The dataset
    """

    # read dataset into dataframes
    df = pd.DataFrame()
    for file in filenames:
        df_temp = pd.read_csv(file, sep='\t', header=None)
        # concat dataframes into a single dataframe
        df = pd.concat([df, df_temp])

    # add a label in each column
    df.columns = mainFields + relatedVideos

    # drop rows with empty values
    df = df.dropna(subset=['id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments'])

    # reset index
    df.reset_index(inplace=True)
    df = df.drop(columns=['index'])

    # write dataset a file
    df.to_csv('dataset.csv', index=False)

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
    df: numpy array
        The dataset with rename id
    """

    # add a new column with a unique integer number to each node
    rename_id = list(range(0, len(df)))
    df['rename_id'] = rename_id

    # reorder columns in dataframe
    columns = []
    columns.append('rename_id')
    columns += mainFields
    columns += relatedVideos
    df = df[columns]

    return df


def create_nodes(df):
    """
    This function will write in a file the informations for each node.

    Parameters
    ----------
    df: numpy array
        The dataset
    """

    # get some columns to another dataframe
    nodes_df = df[['rename_id', 'id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments']]

    # write nodes in a file
    nodes_df.to_csv('nodes.csv', index=False)


def create_edges(df):
    """
    This function will create the edges between nodes and will write them into a file.

    Parameters
    ----------
    df: numpy array
        The dataset
    """

    # create a list with nodes id
    nodes_id = list(df['id'])

    # capture edges in 2 separate lists
    node_list_1 = []
    node_list_2 = []

    # find edges
    for row in tqdm(df.index):
        n1_rename_id = df['rename_id'][row]
        for related in relatedVideos:
            n2_id = df[related][row]
            if n2_id in nodes_id:
                n2_rename_id = df[df['id'] == n2_id]['rename_id'].values[0]
                node_list_1.append(n1_rename_id)
                node_list_2.append(n2_rename_id)

    # insert edges in a dataframe
    edges_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    # write edges a file
    edges_df.to_csv('edges.csv', index=False)

# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#


# read dataset
df = read_data(datasets)

# rename nodes id
df = rename_node(df)

# create nodes
create_nodes(df)

# create edges
create_edges(df)

