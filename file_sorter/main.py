from get_file_embeddings import get_file_embeddings
from clustering import cluster, cluster_indices, create_folder
from sklearn.cluster import DBSCAN, MeanShift, Birch, AffinityPropagation, AgglomerativeClustering
from doc2vec import get_folder_name
# from sklearn import cluster, datasets, mixture
import numpy as np

def demessify():
    file_names, vectors = get_file_embeddings()

    yhat = cluster(DBSCAN, np.stack(vectors))
    print(yhat)
    folders = create_folder(cluster_indices, yhat, file_names)

    return get_folder_name(folders)

print(demessify())