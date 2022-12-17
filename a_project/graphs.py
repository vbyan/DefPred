import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import floor


def make_graph(data, n_neighbors=5, dist=None):
 if data.select_dtypes(['int64', 'int', 'int32', 'object']).shape[1] == data.shape[1]:
  connections = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False, metric='hamming')
 elif data.select_dtypes(['float64', 'float', 'float32']).shape[1] == data.shape[1]:
  connections = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
 elif dist == 'ignore':
  connections = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
 else:
  dist = dist
  connections = kneighbors_graph(dist, n_neighbors, mode='connectivity', include_self=False, metric='precomputed')
 neighbors = pd.DataFrame(connections.toarray())
 graph_source = {}
 for i in range(neighbors.shape[0]):
  graph_source[i] = neighbors.iloc[i, :][neighbors.iloc[i, :] == 1].index.values.tolist()
 G = nx.Graph()
 for key in graph_source.keys():
  for neighbor in graph_source[key]:
   G.add_edge(key, neighbor)
 return G


def subgraphs(graph):
 G = graph
 UG = G.to_undirected()

 # extract subgraphs
 sub_graphs = [(UG.subgraph(c) for c in nx.connected_components(UG))][0]

 subgraph = {}
 for i, sg in enumerate(sub_graphs):
  subgraph[i] = list(sg.nodes())
 return subgraph


class row_selection():
 def __init__(self):
  pass

 def subgraphs(self, data, n_neighbors=1, dist=None):
  G = make_graph(data, n_neighbors=n_neighbors, dist=dist)
  Subgraphs = subgraphs(G)
  print('Number of subgraphs: ' + str(len(Subgraphs.keys())))
  self.Subgraphs = Subgraphs
  self.data = data

 def select(self, percent=10):
  new = []
  for key in self.Subgraphs.keys():
   shape = len(self.Subgraphs[key])
   pick = int(round(shape * percent / 100, 0))
   new_indexes = np.array(self.Subgraphs[key])[np.random.permutation(shape)][:pick]
   for j in new_indexes:
    new += [j]
  return self.data.iloc[new, :]


def batch_row_selection(data, batch_size, percent):
 scaled_data = data.copy()
 scaled_data = pd.DataFrame(StandardScaler().fit_transform(scaled_data), columns=data.columns, index=scaled_data.index)
 indices = np.array([])
 for i in range(floor(data.shape[0] / batch_size) - 1):
  selector = row_selection()
  selector.subgraphs(scaled_data.iloc[batch_size * i:batch_size * (i + 1), :], n_neighbors=2, dist='ignore')
  new = selector.select(percent=percent)
  indices = np.append(indices, new.index.values)
 return data.loc[indices,:]