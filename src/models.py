# petster-hamster-household_edges_class_features.csv

import sys
import pandas as pd
from sklearn import model_selection, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import graphviz

from tabulate import tabulate

def cross_validation(estimator, X, y, score_type, k_folds, num_cpus):
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


def grid_search_cross_validation(clf_list, X, y, score_type='accuracy', k_folds=5, num_cpus=-1):
    """
    Applies grid search to search over specified parameter values for an estimator
    to find the optimal parameters for a machine learning algorithm.
    Also, this function will apply k-folds cross validation to calculate the
    algorithm score in order to select the machine learning algorithm
    with highest score.

    Parameters
    ----------
    clf_list: list of tuples with name of
        Each tuple contains the name of machine learning algorithm, the
        initialization estimator and a set with the parameters
    X: numpy array
        The  data
    y: numpy array
        The labels of data
    score_type: string
        The name of score type
    k_folds: integer
        The number of folds
    num_cpus: int
        The number of cpus that will use this function

    Returns
    -------
    model_names: list of strings
        This list contains the names of machine learning algorithms
    best_estimators: list of estimators
        This list contains the best estimators
    best_parameters: list of dict
        This list contains the best parameters for each machine learning algorithm.
    kfold_accuracy: list of floats
        This list contains the accuracy from k-fold cross validation
    kfold_std: list of floats
        This list contains the standard deviation from accuracy from k-fold cross validation
   """

    model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std = [], [], [], [], []

    print('Start... Grid search and Cross validation')
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)

    for name, model, parameters in clf_list:
        print(name)
        model_names.append(name)

        # grid search
        search = GridSearchCV(model, parameters, scoring=score_type, cv=kfold, n_jobs=num_cpus)
        search.fit(X, y)
        best_parameters.append(search.best_params_)
        best_est = search.best_estimator_  # estimator with the best parameters
        best_estimators.append(best_est)

        # k-fold cross validation
        avg_accuracy, std_accuracy = cross_validation(best_est, X, y, score_type, k_folds, num_cpus)
        kfold_accuracy.append('{:.2f}'.format(avg_accuracy * 100))
        kfold_std.append('{:.2f}'.format(std_accuracy * 100))

    return model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std


def plot_results(x_labels, y_labels, type_name):
    """
    Plots results from k-fold cross validation for each machine learning algorithm.

    Parameters
    ----------
    x_labels: list of strings
        This list contains the names of machine learning algorithms
    y_labels: list of floats
        This list contains the best accuracy or the standard deviation for each algorithm
    type_name: string
        Accuracy or Standard deviation
    """

    plt.figure(figsize=(10, 5))
    plt.title('5-fold cross validation')
    plt.xlabel('machine learning algorithms')
    plt.ylabel(type_name + ' (%)')
    y_labels = [float(x) for x in y_labels]
    plt.bar(x_labels, y_labels)
    plt.savefig('../figures/' + type_name + '.png')
    plt.clf()


def predict_probabilities(X, y, model_names, best_estimators):
    """
    Predict probabilities for each machine learning algorithm.

    Parameters
    ----------
    X: numpy array
        The  data
    y: numpy array
        The labels of data
    model_names: list of strings
        This list contains the names of machine learning algorithms
    best_estimators: list of estimators
        This list contains the best estimators

    Returns
    -------
    y_test: numpy array
        The labels of test data
    pred_prob: list of floats
        This list contains the predicted probabilities for each algorithm
    """

    pred_prob = []

    # split to train (80%) and test (20%) set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, stratify=y)

    for i in range(0, len(model_names)):
        # fitting
        clf = best_estimators[i].fit(X_train, y_train)
        # predict probabilities
        pred_prob.append(clf.predict_proba(X_test))

        # plot decision tree
        if model_names[i] == 'DecisionTree':
            export_graphviz(clf, out_file='../results/tree.dot', class_names=['negative_edge', 'possitive_edge'],
                            feature_names=['jaccard_coef', 'adamic_adar', 'preferential_attachment', 'clustering_coef'],
                            impurity=False, filled=True)
            with open('../results/tree.dot') as f:
                dot_graph = f.read()
            graphviz.Source(dot_graph).view('../results/decision_tree')

        # weights from logistic regression
        if model_names[i] == 'LogisticRegression':
            f = open('../results/logistic_coef.txt', 'w')
            f.write(str(clf.coef_[0]))
            f.close()

    return y_test, pred_prob


def plot_roc_curve(model_names, pred_prob, y_test):
    """
    Plots AUC-ROC curve.

    Parameters
    ----------
    model_names: list of strings
        This list contains the names of machine learning algorithms
    pred_prob: list of floats
        This list contains the predicted probabilities for each algorithm
    y_test: numpy array
        The labels of test data

    Returns
    -------
    auc_list: list of floats
        This list contains AUC scores
    """

    auc_list = []
    plt.figure()
    # define the color list fot ruc curve
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'gray']
    for i, esti in enumerate(model_names):
        # roc curve for models
        fpr, tpr, thresh = roc_curve(y_test, pred_prob[i][:, 1], pos_label=1)
        # auc scores
        auc_score = roc_auc_score(y_test, pred_prob[i][:, 1])
        auc_list.append('{:.2f}'.format(auc_score * 100))
        # plot roc curves
        plt.plot(fpr, tpr, color=colors[i], label=esti + ' (AUC=' + '{:.2f}'.format(auc_score*100) + '%)')

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend()
    plt.savefig('../figures/roc.png')
    plt.clf()

    return auc_list


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

# create list with all possible parameters for each estimator
"""
clf_list = [('LogisticRegression', LogisticRegression(), {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                                                          'max_iter': [100, 500, 1000]}),
            ('kNN', KNeighborsClassifier(), {'n_neighbors': [5, 10, 15, 20],
                                             'metric': ['euclidean', 'minkowski', 'manhattan']}),
            ('MLP', MLPClassifier(), {'activation': ['tanh', 'relu'],
                                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                      'max_iter': [200, 500, 1000]}),
            ('DecisionTree', DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'],
                                                        'splitter': ['best', 'random'],
                                                        'max_features': ['auto', 'sqrt', 'log2']}),
            ('RandomForest', RandomForestClassifier(), {'n_estimators': [100, 500, 1000],
                                                        'criterion': ['gini', 'entropy'],
                                                        'max_features': ['auto', 'sqrt', 'log2']})]
"""
clf_list = [('LogisticRegression', LogisticRegression(), {}),
            ('kNN', KNeighborsClassifier(), {}),
            ('MLP', MLPClassifier(), {'activation': ['tanh'],
                                      'learning_rate': ['constant'],
                                      'max_iter': [200]}),
            ('DecisionTree', DecisionTreeClassifier(), {}),
            ('RandomForest', RandomForestClassifier(), {}),
            ('GaussianNB', GaussianNB(), {})]

# clf_list = [('LogisticRegression', LogisticRegression(), {})]
# grid search and cross validation
model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std = grid_search_cross_validation(clf_list, X, y)

# print results
header_label = ['Model', 'Accuracy', 'Std', 'BestParameters']

# plot results from cross validation
plot_results(model_names, kfold_accuracy, 'accuracy')
plot_results(model_names, kfold_std, 'standard_deviation')

# predict probabilities
y_test, pred_prob = predict_probabilities(X, y, model_names, best_estimators)

# plot ROC AUC curve
auc_list = plot_roc_curve(model_names, pred_prob, y_test)

df = pd.DataFrame({'Model': model_names, 'Accuracy(%)': kfold_accuracy, 'Std(%)': kfold_std, 'AUC(%)': auc_list, 'BestParameters': best_parameters})
print(tabulate(df, headers='keys', showindex=False))
