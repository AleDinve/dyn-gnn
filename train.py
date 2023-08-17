from torch_geometric.nn import GraphConv, global_add_pool
import torch
from torch.nn import RNN, Linear
import torch.nn.functional as F
from model import DYN_GNN

from wl_test import dyn_wl_generator

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
card_set = [70,30]

num_nodes = 3
num_it = 10
seq_len = 2
train_loader, test_loader = dyn_wl_generator(card_set, num_nodes, num_it, seq_len)


lr = 0.0001
model = DYN_GNN(input_gnn=1, hidden_gnn=32, output_gnn=8, layers_gnn=4, 
                 hidden_rnn=8, sequence_length=seq_len, device = device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 500

@torch.enable_grad()
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        #num_nodes = int(data[0].num_nodes/data[0].num_graphs)
        pred = model(data, data[0].num_graphs)
        loss = F.mse_loss(pred, data[0].y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data[0].num_graphs
    return total_loss / len(train_loader.dataset[0])

@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        pred = model(data, data[0].num_graphs)          
        loss = F.mse_loss(pred, data[0].y)
        total_correct += int((torch.abs(pred-data[0].y)<0.1).sum())
    return total_correct * 100 / len(loader.dataset[0]), loss.item()

for epoch in range(1, epochs + 1):
    print(f'epoch {epoch}')
    train_l = train()     
    train_acc, train_loss = test(train_loader)
    test_acc, test_loss = test(test_loader)
    print(f'Train loss:{train_l}')
    print(f'Train accuracy: {train_acc}, train loss: {train_l}')
    print(f'Test accuracy: {test_acc}, test loss: {test_loss}')
