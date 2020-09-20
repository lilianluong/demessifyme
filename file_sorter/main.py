#! python
from file_read_write import get_file_embeddings, write_folders
from clustering import cluster, cluster_indices, create_folder
from sklearn.cluster import DBSCAN, MeanShift, Birch, AffinityPropagation, AgglomerativeClustering
from doc2vec import get_folder_name
# from sklearn import cluster, datasets, mixture
import numpy as np
import os


def demessify():
    file_names, vectors = get_file_embeddings()

    yhat = cluster(DBSCAN, np.stack(vectors))
    print("Sorted files.")
    folders = create_folder(cluster_indices, yhat, file_names)
    named_folders = get_folder_name(folders)
    write_folders(named_folders)
    print("Done. Created", len([1 for f in named_folders.values() if len(f)]), "folders.")


if __name__ == "__main__":
    demessify()
