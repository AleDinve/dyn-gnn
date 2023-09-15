'''@author: Giuseppe Alessio D'Inverno'''
'''@date: 02/03/2023'''

import torch
from torch_geometric import seed
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from itertools import product
import numpy as np
from utils import  *
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed.seed_everything(10)
## WL class to find colors

def dataset_generator(card_set, num_nodes, batch_size=1, save =False):
    dataset = []
    for i in range(card_set):
        A = np.random.randint(0,50,(num_nodes, num_nodes))
        
        #print(A)
        A = (A > 20).astype(int)
        np.fill_diagonal(A,0)
        if save:
            np.savetxt('graphs/graph'+str(i)+'.txt',A.astype(int))
        
        G = nx.from_numpy_matrix(A).to_undirected()
        g = from_networkx(G)
        g.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
        dataset.append(g)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    return data_loader

def dyn_dataset_generator(card_set, seq_len, num_nodes, batch_size=1, save =False, labels = None,):
    dataset = [[],[]]
    loader = []

    for j in range(len(card_set)):
        for i in range(card_set[j]):
            
            dyn_graph = []
            for t in range(seq_len):
                #random graph generation
                A = np.random.randint(0,50,(num_nodes, num_nodes))
                A = (A > 40).astype(int) 
                np.fill_diagonal(A,0)

                #if isolated nodes exist, re-generate the graph
                while np.min(np.sum(A,axis=0)-np.ones(num_nodes))<0:
                    A = np.random.randint(0,50,(num_nodes, num_nodes))
                    A = (A > 20).astype(int)
                    np.fill_diagonal(A,0)

                # A1, A2 = wl_equiv_graphs()

                # if np.amax(np.abs(A-A1))<1:
                #     print('trovato1')
                # if np.amax(np.abs(A-A2))<1:
                #     print('trovato2')
                
                if save:
                    np.savetxt('graphs/graph'+str(i)+'_time_'+str(t)+'.txt',A.astype(int))
                
                G = nx.from_numpy_matrix(A).to_undirected()
                g = from_networkx(G)
                g.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
                if labels is not None:
                    g.y = labels[j][i]
                dyn_graph.append(g)

            dataset[j].append(dyn_graph)
    loader.append(DataLoader(dataset[0], batch_size, shuffle=False))
    loader.append(DataLoader(dataset[1], batch_size, shuffle=False))
    return loader, dataset

def dyn_dataset_relabel(dataset, card_set, seq_len, num_nodes, labels, batch_size=1):
    for j in range(len(card_set)):
        for i in range(card_set[j]):
            for t in range(seq_len):
                dataset[j][i][t].y = labels[j][i]
    train_loader = DataLoader(dataset[0], batch_size, shuffle=False)
    test_loader = DataLoader(dataset[1], batch_size, shuffle=False)
    return train_loader, test_loader


class WL(torch.nn.Module):
    def __init__(self,  num_it):
        super().__init__()
        self.num_it = num_it
        self.conv = WLConv()

    def forward(self, x, edge_index):
        for _ in range(self.num_it):
            x = self.conv(x, edge_index)   
        return x

def wl_colors(model, G, hashing = True):
    
    pred = model(G.x, G.edge_index)
    if hashing:
        pred = hash(tuple(sorted(pred.tolist())))
    else:
        pred, _, count = torch.unique(pred, return_inverse= True, return_counts = True)
        pred = torch.stack([pred,count])
    return pred

def dyn_wl(model, G, hashing = True):
    seq = []
    for t in range(len(G)):
        col = wl_colors(model, G[t], hashing = hashing)
        seq.append(col)
    pred = hash(tuple(seq))
    return pred

def dyn_wl_generator(card_set, num_nodes, num_it, seq_len):

    model = WL(num_it).to(device)
    model.eval()
    loader, dataset = dyn_dataset_generator(card_set, seq_len, num_nodes, batch_size=1)
    batchdata_list = []
    color_list = []
    for j in range(len(card_set)):
        color_list.append([])
        for data in loader[j]:
            batchdata_list.append(data)
            col = dyn_wl(model, data, True)
            #print(col)
            color_list[j].append(col)
        #color_list[j] = to_categorical(color_list[j])
        amin, amax = min(color_list[j]), max(color_list[j])
        for i, val in enumerate(color_list[j]):
            color_list[j][i] = (val-amin) / (amax-amin)
    #train_loader, test_loader = dyn_dataset_generator(card_set, seq_len, num_nodes, batch_size=1, save=False, labels=color_list)
    train_loader, test_loader = dyn_dataset_relabel(dataset, card_set, seq_len, num_nodes, labels=color_list, batch_size=1)
    return train_loader, test_loader





def main1():
    card_set = [120,40]
    num_nodes = 6
    num_it = 10
    seq_len = 2
    train_loader, test_loader = dyn_wl_generator(card_set, num_nodes, num_it, seq_len)

def dyn_graphs1():
    num_it = 10
    num_nodes = 6
    A1, A2 = wl_equiv_graphs()
    # A1 = cycle_graph((6))
    # A2 = triangles()
    model = WL(num_it).to(device)
    model.eval()
    G1 = nx.from_numpy_matrix(A1).to_undirected()
    g1 = from_networkx(G1)
    g1.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    G2 = nx.from_numpy_matrix(A2).to_undirected()
    g2 = from_networkx(G2)
    g2.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)

    return g1,g2
    # print(dyn_wl(model, d1))
    # print(dyn_wl(model, d2))
    # print(dyn_wl(model, d3))
    # print(dyn_wl(model, d4))

def dyn_graphs2():
    num_it = 10
    num_nodes = 6
    # A1, A2 = wl_equiv_graphs()
    A1 = cycle_graph((6))
    A2 = triangles()
    model = WL(num_it).to(device)
    model.eval()
    G1 = nx.from_numpy_matrix(A1).to_undirected()
    g1 = from_networkx(G1)
    g1.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    G2 = nx.from_numpy_matrix(A2).to_undirected()
    g2 = from_networkx(G2)
    g2.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    return g1,g2

def dyn_synth():
    train_dataset = []
    test_dataset = []
    for i in range(12):
        train_dataset += dyn_graphs1()
        train_dataset += dyn_graphs2()
    for i in range(2):
        test_dataset += dyn_graphs1()
        test_dataset += dyn_graphs2()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, test_loader



def dyn_synth_prod(seq_len, num_it = 7, batch_size = 1):
    model = WL(num_it).to(device)
    model.eval()
    graphs = []
    graphs += dyn_graphs1()
    graphs += dyn_graphs2()
    indices = product([0,1,2,3], repeat=seq_len)
    dyn_graph_list = []
    color_list = []
    for elem in indices:
        dyn = []
        for i in range(seq_len):
            dyn.append(graphs[elem[i]])
        col = dyn_wl(model, dyn, True)
        dyn_graph_list.append(dyn)
        color_list.append(col)
    print('checkpoint')
    amin, amax = min(color_list), max(color_list)
    for i, val in enumerate(color_list):
        color_list[i] = (val-amin) / (amax-amin)
    for i, g  in enumerate(dyn_graph_list):
        for t in range(seq_len):
            g[t].y = color_list[i]
        dyn_graph_list[i] = g
    train_loader = DataLoader(dyn_graph_list, batch_size=batch_size, shuffle=True)
    return train_loader


if __name__ == '__main__':
    train_loader = dyn_synth_prod(5)
    print(train_loader.dataset[0])

