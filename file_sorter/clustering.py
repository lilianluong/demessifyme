import numpy as np
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift, DBSCAN, Birch
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
from numpy import unique
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# create dataset with 5 centers and 1000 points
from sklearn.datasets import make_blobs


def create_data(n_samples, centers):
    """
    :param n_samples: number of points in dataset
    :param centers: numbers of centers for the groups
    :return: X (2D numpy array of a list of vectors) and y (numpy array of class labels)
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=15)
    return X, y


X, y = create_data(1000, 5)


def plot_data(centers, X, y):
    """
    :param centers: numbers of centers or groups in data
    :param X: 2D numpy array of a list of vectors
    :param y: numpy array of class labels
    :return: none, plots samples from each class in a scatter plot
    """
    # create scatter plot for samples from each class
    for class_value in range(centers):
        # get row indexes for samples with this class
        row_ix = where(y == class_value)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


plot_data(5, X, y)


def test_clustering(alg, X, y):
    """
    :param alg: name of clustering algorithm (e.g. DBSCAN)
    :param X: 2D numpy array of a list of vectors
    :param y: numpy array of class labels
    :return: list of labels
    """
    # define the model
    model = alg()
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()
    return yhat


# uses DBSCAN clustering algorithm: density based with
# eps (epsilon) as most important parameter 
test_clustering(DBSCAN, X, y)


def cluster_indices(label, yhat):
    """
    :param label: specified label or class (e.g. 3)
    :param yhat: numpy array of class labels
    :return: numpy array of list of indices where the specified label = actual label (yhat)
    """
    return np.where(label == yhat)[0]


def create_folder(cluster_indices, yhat):
    """
    :param cluster_indices: function that returns numpy array of list of indices where the specified label = actual label (yhat)
    :param yhat: numpy array of class labels
    :return: list of list of indices by label
    """
    folders = [cluster_indices(-1, yhat)]
    for i in range(max(yhat) + 1):
        print(i)
        folders.append(cluster_indices(i, yhat))
    return folders


yhat = test_clustering(DBSCAN, X, y)
folders = create_folder(cluster_indices, yhat)


def find_means(folders):
    """
    :param folders: list of list of indices by label
    :return: list of mean values of vectors in each folder
    """
    means = []
    for i in range(len(folders)):
        means.append(np.mean(folders[i]))
    return means


means = find_means(folders)
print(means)
