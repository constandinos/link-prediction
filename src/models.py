import sys
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def cross_validation(estimator, X, y, score_type='accuracy', k_folds=5, num_cpus=15):
    """
    Applies k-folds cross validation to calculate the algorithm score in order to select
    the machine learning algorithm with highest score.

    Parameters
    ----------
    estimator: estimators
        Estimator (ml or nn) algorithm
    X: numpy array
        The  data
    y: numpy array
        The labels of data
    score_type: str
        The name of score type
    k_folds: int
        The number of folds
    num_cpus: int
        The number of cpus that will use this function

    Returns
    -------
    avg_accuracy: float
        The cross validation accuracy of machine learning algorithm
    std_accuracy: float
        The cross validation standard deviations of machine learning algorithms
    """

    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # k-fold cross validation
    score = model_selection.cross_val_score(estimator, X, y.values.ravel(), cv=kfold, scoring=score_type, n_jobs=num_cpus)
    # append results to the return lists
    avg_accuracy = score.mean()
    std_accuracy = score.std()

    return avg_accuracy, std_accuracy


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define input files
filein = sys.argv[1]  # eg. facebook-wosn-links_edges_class_features.csv

# read input data
df_features = pd.read_csv('../dataset/' + filein)
print('Edges with features:')
print(df_features)
print()

# get data (X) and labels (Y)
X = df_features.drop(['source_node', 'destination_node', 'class'], axis=1)
y = df_features[['class']]
print('Data:')
print(X)
print()
print('Lables:')
print(y)
print()

# standardize features by removing the mean and scaling to unit variance
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

clf = LogisticRegression(random_state=0, n_jobs=15)
print(cross_validation(clf, X, y))
