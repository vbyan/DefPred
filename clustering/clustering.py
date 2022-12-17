import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE as tsne
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans,Birch
from sklearn.mixture import GaussianMixture as GMM
from validclust import dunn
from s_dbw import S_Dbw
from sklearn.metrics.pairwise import pairwise_distances
from math import ceil, floor
from sklearn.decomposition import KernelPCA
from kmodes.kmodes import KModes
from gower import gower_matrix
from CFS import helpers
from kmodes.kprototypes import KPrototypes

def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b)


def locate_categoricals(data):
    categorical_indices = []
    categorical_columns  = data.select_dtypes(['int','int64','object','int32']).columns.values
    for column in categorical_columns:
        categorical_indices += [np.where(data.columns.values == column)[0][0]]
    return categorical_indices

def weights_for_gower(data):
    categorical_indices = locate_categoricals(data)
    cat_weight = data.drop(data.iloc[:,categorical_indices].columns, axis = 1).values.reshape(-1).std() * 0.5
    base_array = np.repeat(1,data.shape[1]).astype('float32')
    base_array[categorical_indices] = cat_weight
    weights = base_array
    return weights

def numerical_metric_validation(data, max_clusters, title = 'a',include = 'all', algorithm = 'GMM',min_clusters = 2):

    std = data
    K = range(min_clusters, max_clusters)
    metric_dict = {
    's' : [silhouette_score,'max'],
    'S' : [silhouette_score,'max'],
    'db': [davies_bouldin_score,'min'],
    'DB': [davies_bouldin_score,'min'],
    'ch': [calinski_harabasz_score,'max'],
    'CH': [calinski_harabasz_score,'max'],
    'd' : [dunn,'max'],
    'D' : [dunn,'max'],
    'sd': [S_Dbw,'min'],
    'SD': [S_Dbw,'min']
    }
    if algorithm == 'GMM':
        models = [GMM(k, covariance_type='full', random_state=116).fit(std)
                  for k in K]
    elif algorithm == 'KMeans':
        models = [KMeans(n_clusters=k, random_state=42).fit(std) for k in K]
    elif algorithm == 'Birch':
        models = [Birch(n_clusters=k,random_state = 42).fit(std) for k in K]


    if include == 'all':
        include = ['S','DB','CH','D']
        functions = []
        for key in include:
            functions += [metric_dict[key][0]]
        n_metrics = len(include)
        for metric in functions:
            if metric == dunn:
                dist = pairwise_distances(std)
                np.fill_diagonal(dist,0)
                globals()[str(eval('metric')).split(' ')[1] + 's'] = [metric(dist, model.predict(std)) for model in
                                                                      models]
            elif metric == silhouette_score:
                dist = pairwise_distances(std)
                np.fill_diagonal(dist,0)
                globals()[str(eval('metric')).split(' ')[1] + 's'] = [metric(dist, model.predict(std), metric = 'precomputed') for model in
                                                                      models]
            else:
                globals()[str(eval('metric')).split(' ')[1] + 's'] = [metric(std, model.predict(std)) for model in
                                                                      models]
    else:
        n_metrics = len(include)
        functions = []
        for key in include:
            functions += [metric_dict[key][0]]
        for metric in functions:
            if metric == dunn:
                dist = pairwise_distances(std)
                globals()[str(eval('metric')).split(' ')[1] + 's'] = [metric(dist, model.predict(std)) for model in
                                                                      models]
            else:
                globals()[str(eval('metric')).split(' ')[1] + 's'] = [metric(std, model.predict(std)) for model in
                                                                      models]

    plt.figure(figsize=(15, 4 * ceil(n_metrics/2)))
    plt.suptitle(title + ' - ' + algorithm)
    metric_lists = {}
    for metric, metric_name, index in zip(functions,include,np.arange(1,len(include)+1)):
        metric_list =  eval(eval("str(metric).split(' ')[1] + 's'"))
        if metric_dict[metric_name][1] == 'min':
            metric_lists[metric_name] = (max(metric_list) / np.array(metric_list)).tolist()
        else:
            metric_lists[metric_name] = metric_list
        plt.subplot(ceil(n_metrics / 2), 2, index)
        plt.plot(K, eval(eval("str(metric).split(' ')[1] + 's'")), markersize=10, c='r')
        plt.title(metric_name, fontsize=15)
        plt.grid()
    return metric_lists


def categorical_metric_validation(data, max_clusters, title = 'a',include = 'all', algorithm = 'KModes',  min_clusters = 1, gamma = None,
                                  max_iter = 100, n_init = 10, init = 'Huang'):
    def inertia(model):
        return model.cost_
    std = data
    K = range(min_clusters, max_clusters)
    metric_dict = {
        's': [silhouette_score, 'max'],
        'S': [silhouette_score, 'max'],
        'db': [davies_bouldin_score, 'min'],
        'DB': [davies_bouldin_score, 'min'],
        'ch': [calinski_harabasz_score, 'max'],
        'CH': [calinski_harabasz_score, 'max'],
        'd': [dunn, 'max'],
        'D': [dunn, 'max'],
        'I': [inertia, 'max'],
        'i': [inertia, 'max']
    }
    if algorithm == 'KModes':
        models = [KModes(k, random_state=42, n_jobs = -1).fit(std)
                  for k in K]
        dist = pairwise_distances(std.values, metric= matching_dissim,  n_jobs = -1)
        np.fill_diagonal(dist, 0)
    elif algorithm == 'k_prototypes':
        categorical_indices = locate_categoricals(std)
        models = [KPrototypes(k, random_state=42, n_jobs = -1, gamma = gamma,max_iter = 100, n_init = 10,
                              init = 'Huang').fit(std, categorical = categorical_indices)
                  for k in K]
        weight = weights_for_gower(data)
        is_number = np.vectorize(lambda x: not (np.issubdtype(x, 'float32') | np.issubdtype(x, 'float64')))
        cat_features = is_number(std.dtypes)
        dist = gower_matrix(std.values, weight = weight, cat_features = cat_features)
        np.fill_diagonal(dist, 0)


    if include == 'all':
        include = ['S','D','DB','CH','I']
        functions = []
        for key in include:
            functions += [metric_dict[key][0]]
        n_metrics = len(include)
        for metric in functions:
            if metric == dunn:
                dunns = [metric(dist, model.labels_) for model in
                                                                      models[1:]]
            elif metric == silhouette_score:
                silhouette_scores = [metric(dist, model.labels_, metric='precomputed') for model in
                                                                      models[1:]]
            elif metric == davies_bouldin_score:
                davies_bouldin_scores = [metric(std, model.labels_, metric= 'gower') for model in
                                                                      models[1:]]
            elif metric == calinski_harabasz_score:
                calinski_harabasz_scores = [metric(std, model.labels_) for model in
                                                                      models[1:]]
            else:
                inertias = [metric(model) for model in models]

    else:
        n_metrics = len(include)
        functions = []
        for key in include:
            functions += [metric_dict[key][0]]
        for metric in functions:
            if metric == dunn:
                dunns = [metric(dist, model.labels_) for model in
                                                                      models[1:]]
            elif metric == silhouette_score:
                silhouette_scores = [metric(dist, model.labels_, metric='precomputed') for model in
                                                                      models[1:]]
            elif metric == davies_bouldin_score:
                davies_bouldin_scores = [metric(std, model.labels_, metric='hamming') for model in
                                                                      models[1:]]
            elif metric == calinski_harabasz_score:
                calinski_harabasz_scores = [metric(std, model.labels_) for model in
                                                                      models[1:]]
            else:
                inertias = [metric(model) for model in models]

    plt.figure(figsize=(15, 4 * ceil(n_metrics / 2)))
    plt.suptitle(title + ' - ' + algorithm)
    metric_lists = {}
    for metric, metric_name, index in zip(functions, include, np.arange(1, len(include) + 1)):
        if metric_name == 'I':
            metric_list = eval(str(metric).split('.')[2].split(' ')[0] + 's')
        else:
            metric_list = eval(str(metric).split(' ')[1] + 's')
        if metric_dict[metric_name][1] == 'min':
            metric_lists[metric_name] = (max(metric_list) / np.array(metric_list)).tolist()
        else:
            metric_lists[metric_name] = metric_list
        plt.subplot(ceil(n_metrics / 2), 2, index)
        plt.plot(range(1,len(metric_list)+1), metric_list, markersize=10, c='r')
        plt.title(metric_name, fontsize=15)
        plt.grid()
    return metric_lists


def decomposition(data, percent = 100, labels = ['g']):
    dimension = data.shape[1]
    size = int(format(round(len(data.index) * percent / 100, 0), '.0f'))
    data = data.iloc[:size, :]
    labels = labels[:size]
    std = data
    n_components = int(projection[0])
    model = KernelPCA(n_components=n_components, kernel='sigmoid', gamma=0.01).fit(std)
    reduced = pd.DataFrame(model.transform(std))
    if n_components == 2:
        plt.figure(figsize=(15, 7))
        plt.scatter(reduced.iloc[:, 0], reduced.iloc[:, 1], c=labels)
        plt.title(
            method + '\n Number of features = ' + str(dimension) + '\n' + str(pd.Series(labels).value_counts())[
                                                                          :-13])
        plt.show()
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced.iloc[:, 0], reduced.iloc[:, 1], reduced.iloc[:, 2], c=labels)
        plt.title(
            method + '\n Number of features = ' + str(dimension) + '\n' + str(pd.Series(labels).value_counts())[
                                                                          :-13])
        plt.show()



class TSNE():
    def __init__(self):
        pass
    def transform(self,data,perplexity, metric = 'euclidean'):
        Trans = []
        std = data
        if type(perplexity) is list:
            for index,perp in enumerate(perplexity):
                model = tsne(n_components=2, random_state=42, perplexity=perp, metric = metric)
                transformed = pd.DataFrame(model.fit_transform(std))
                Trans += [transformed]
        else:
            model = tsne(n_components=2, random_state=42, perplexity=perplexity, metric = metric)
            Trans = pd.DataFrame(model.fit_transform(std))
        self.Trans = Trans
        self.perp = perplexity
        return self.Trans
    def visualize(self,labels = ['g']):
        if type(self.Trans) is list:
            shape = len(self.Trans)
            plt.figure(figsize=(15, 4 * int(np.ceil(shape / 2))))
            for trans,perp,index in zip(self.Trans,self.perp,np.arange(shape)):
                plt.subplot(int(np.ceil(shape / 2)), 2, index + 1)
                plt.scatter(trans.iloc[:, 0], trans.iloc[:, 1], c=labels)
                plt.title('Perplexity = ' + str(perp), fontsize=12)
        else:
            plt.figure(figsize = (15,7))
            plt.scatter(self.Trans.iloc[:, 0], self.Trans.iloc[:, 1], c=labels)


def GMM_model(data, n_clusters):
    std = data
    model_gmm = GMM(n_clusters, covariance_type='full', random_state=116).fit(std)
    return model_gmm


def KMeans_model(data, n_clusters):
    std = data
    model_kms = KMeans(n_clusters=n_clusters, random_state=42).fit(std)
    return model_kms

def KModes_model(data,n_clusters):
    std = data
    model_kms = KModes(n_clusters=n_clusters, random_state=42, n_jobs = -1).fit(std)
    return model_kms

def KPrototypes_model(data,n_clusters, gamma = None, max_iter = 100, n_init = 10, init = 'Cao'):
    std = data
    categorical_indices = locate_categoricals(std)
    model_kprts = KPrototypes(n_clusters=n_clusters, random_state=42, n_jobs = -1, gamma = gamma, max_iter = max_iter, n_init = n_init, init = init).fit(std, categorical = categorical_indices)
    return model_kprts

def manhattan_dissim(a, b, **_):
    """Manhattan distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    try:
        return abs(np.sum(a - b, axis=1))
    except:
        return abs(np.sum(a - b))