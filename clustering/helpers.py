import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import inspect
from .fun import *
from math import floor,ceil
from  math import factorial
import multiprocessing.pool as mpp


def nan_values(data, axis = 0):
    nans = []
    if axis == 0:
        for column in data.columns:
            nans += [data[column].shape[0] - data[column].dropna().shape[0]]
        nans = pd.Series(nans, index = data.columns)
        return nans
    elif axis == 1:
        for row in data.index:
            nans += [data.loc[row,:].shape[0] - data.loc[row,:].dropna().shape[0]]
        nans = pd.Series(nans, index = data.index)
        return nans


def categorical_encoder(data,columns):
    for column in columns:
        unique_values = data[column].unique().tolist()
        dict = {}
        for i,j in enumerate(unique_values):
            dict[j] = i
        data[column] = data[column].replace(dict)
    return data

class row_selection():
    def __init__(self):
        pass

    def subgraphs(self, data, n_neighbors = 1):
        G = make_graph(data, n_neighbors=n_neighbors)
        Subgraphs = subgraphs(G)
        print('Number of subgraphs: ' + str(len(Subgraphs.keys())))
        self.Subgraphs = Subgraphs
        self.data = data

    def select(self, percent = 10):
        new =[]
        for key in self.Subgraphs.keys():
            shape = len(self.Subgraphs[key])
            pick = int(round(shape * percent / 100,0))
            new_indexes = np.array(self.Subgraphs[key])[np.random.permutation(shape)][:pick]
            for j in new_indexes:
                new += [j]
        return self.data.iloc[new, :]



def window_scaler(data, window_size = [2,3,5]):
    Scores = pd.DataFrame([])
    for window in  window_size:
        starter = Scores.shape[1]
        for i in range(data.shape[1] - window + 1):
            sample = data.iloc[:,i:i+window]
            sample = pd.DataFrame(sample.values, columns=sample.columns)
            sample['score'] = StandardScaler().fit_transform(sample.sum(axis = 1).values.reshape(-1,1))
            Scores[str(i+starter)] = sample['score']
    return Scores

def best_thresh(data):
    scores = []
    Range = range(data.shape[1] - 100,data.shape[1])
    for i in Range:
        temp = data.dropna(thresh = i, axis = 0)
        final = selector(temp, thresh1 = 40, thresh2 = 40)
        scores += [final.shape[0]]
    best_thresh = pd.Series(scores, index = Range).idxmax()
    return best_thresh

def selector(data, thresh1 = 50, thresh2 = 50, row_thresh = 'auto'):
    if row_thresh == 'auto':
        thresh = best_thresh(data)
    else:
        thresh = row_thresh
    data = data.dropna(thresh = thresh, axis = 0)
    columns  = pd.Series(data.columns.values)
    columns_1 = columns[~((columns.str.startswith('Q'))&(columns.str.contains('\\d')))].values.tolist()
    columns_2 = columns[((columns.str.startswith('Q'))&(columns.str.contains('\\d')))].values.tolist()
    nan_values_1 =nan_values(data.loc[:,columns_1])
    nan_values_2 = nan_values(data.loc[:,columns_2])
    out_columns_1 = nan_values_1.sort_values(ascending = False)[:thresh1].index.values.tolist()
    out_columns_2 = nan_values_2.sort_values(ascending = False)[:thresh2].index.values.tolist()
    selected_columns_1 = set(columns_1) - set(out_columns_1)
    selected_columns_2 = set(columns_2) - set(out_columns_2)
    selected_columns = list(selected_columns_1 | selected_columns_2)
    return data.loc[:,selected_columns].dropna()

def get_arguments(iterables, constants, n_jobs, function):
    runners = []
    for i in range(n_jobs):
        runners += [list()]
    signature = inspect.signature(function)
    all_args = [k[0] for k  in signature.parameters.items()]
    all_args_with_values = { k: v.default for k, v in signature.parameters.items()}
    Iterables = {}
    Constants = {}
    for i in iterables:
        try:
            Iterables[i] = pd.Series(
                str(input(str(n_jobs) + ' ' + 'Values for iterable variable' + '  ' + str(i) + ':  ')).split(
                    ';')).values
        except:
            Iterables[i] = pd.Series(
                str(input(str(n_jobs) + ' ' + 'Values for iterable variable' + '  ' + str(i) + ':  ')).split(
                    ';')).values
    for i in constants:
        Constants[i] = str(input('Value for constant variable' + '  ' + str(i) + ':  '))
    for key in all_args:
        if key in Iterables:
            for k,runner in enumerate(runners):
                runner += [Iterables[key][k]]
        elif key in Constants:
            for runner in runners:
                runner += [Constants[key]]
        else:
            for runner in runners:
                runner += [all_args_with_values[key]]
    Runners = []
    for runner in runners:
        Runners += [tuple(runner)]
    with open("Runners.txt", "wb") as fp:
        pickle.dump(Runners,fp)
    with open("Runners.txt", "rb") as fp:
        Runners = pickle.load(fp)
    return Runners


def C(n,m):
    return int(factorial(n)/(factorial(m) * factorial(n-m)))

