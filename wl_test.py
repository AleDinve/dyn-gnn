'''@author: Giuseppe Alessio D'Inverno'''
'''@date: 02/03/2023'''

import torch
from torch_geometric import seed
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import numpy as np
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
    train_dataset = []
    test_dataset = []
    for j in range(2):
        for i in range(card_set[j]):
            dyn_graph = []
            for t in range(seq_len):
                A = np.random.randint(0,50,(num_nodes, num_nodes))
                
                #print(A)
                A = (A > 20).astype(int)
                
                np.fill_diagonal(A,0)
                if save:
                    np.savetxt('graphs/graph'+str(i)+'_time_'+str(t)+'.txt',A.astype(int))
                
                G = nx.from_numpy_matrix(A).to_undirected()
                g = from_networkx(G)
                g.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
                if labels is not None:
                    g.y = labels[j][i]
                dyn_graph.append(g)
            if j==0:
                train_dataset.append(dyn_graph)
            else:
                test_dataset.append(dyn_graph)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
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
    loader = dyn_dataset_generator(card_set, seq_len, num_nodes, batch_size=1)
    batchdata_list = []
    color_list = []
    for j in range(2):
        color_list.append([])
        for data in loader[j]:
            batchdata_list.append(data)
            col = dyn_wl(model, data, True)
            #print(col)
            color_list[j].append(col)
        amin, amax = min(color_list[j]), max(color_list[j])
        for i, val in enumerate(color_list[j]):
            color_list[j][i] = (val-amin) / (amax-amin)
    print(color_list)

    train_loader, test_loader = dyn_dataset_generator(card_set, seq_len, num_nodes, batch_size=1, save=False, labels=color_list)

    return train_loader, test_loader
    #print(np.unique(color_list).size)


if __name__ == '__main__':
    card_set = [80,20]
    num_nodes = 3
    num_it = 10
    seq_len = 2
    train_loader, test_loader = dyn_wl_generator(card_set, num_nodes, num_it, seq_len)
