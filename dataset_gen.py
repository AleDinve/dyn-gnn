
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from string import ascii_uppercase as letters
import networkx as nx
import random

def cycle_graph(n):

    A = np.zeros((n,n))
    A[0,-1]=1
    A[-1,0]=1
    for i in range(n-1):
        A[i,i+1]=1
        A[i+1,i]=1
    s = np.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    g = from_networkx(G)
    g.x = torch.tensor([[1] for i in range(s)], dtype=torch.float)
    g.y = torch.tensor([[1]],dtype=torch.float)
    return g

def temporal_graph(n, k):
    x = []
    for _ in range(k):
        x.append(cycle_graph(n))
    return x

def dataset(num_graphs, N, k):
    dataset_list = []
    for _ in range(num_graphs):
        dataset_list.append(temporal_graph(N,k))
    random.shuffle(dataset_list)
    dataloader = DataLoader(dataset_list, num_graphs, shuffle=False)
    return dataloader

if __name__ == '__main__':
    x = temporal_graph(5,3)
    print(x[0].x)
    data = dataset(5,3,3)
    for item in data:
        print(len(item))
        for i in range(len(item)):
            x = item[i].x
            print(x)
    