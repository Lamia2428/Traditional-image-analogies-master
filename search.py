from parameters import search_method as method
from sklearn.neighbors import NearestNeighbors
import pyflann as pyflann

def nearest_neighbor_indexes(source_f_vect):
    if method == 'pyflann_kmeans':
        flann = pyflann.FLANN()
        index = flann.build_index(source_f_vect, algorithm="kmeans", branching=32, iterations=-1, checks=16)
        return index, flann
    elif method == 'pyflann_kdtree':
        flann = pyflann.FLANN()
        index = flann.build_index(source_f_vect, algorithm="kdtree")
        return index, flann
    elif method == 'sk_nn':
        index = NearestNeighbors(n_neighbors=1, algorithm='kd_tree',metric='l2',n_jobs=-1).fit(source_f_vect)
        return index, None
    else:
        raise ValueError('method not recognized')


def query_neighbors(index,target_f_vect,flann=None):
    if method == 'pyflann_kmeans':
        neighbors, distances = flann.nn_index(target_f_vect, 1, checks=index['checks'])
        return neighbors, distances
    elif method == 'pyflann_kdtree':
        neighbors, distances = flann.nn_index(target_f_vect, 1, checks=index['checks'])
        return neighbors, distances
    elif method == 'sk_nn':
        distances, neighbors = index.kneighbors(target_f_vect)
        return neighbors, distances
    else:
        raise ValueError('method not recognized')        

