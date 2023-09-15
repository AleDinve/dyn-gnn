from torch_geometric.nn import GraphConv, global_add_pool
import torch
from torch.nn import RNN, Linear
import torch.nn.functional as F
from model import DYN_GNN
import pandas as pd

from wl_test import dyn_wl_generator, dyn_synth_prod

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def model_training(num_nodes, seq_len, hidden_gnn):
    raw_data = []
    train_loader = dyn_synth_prod(seq_len, batch_size=32)
    # eval_train_loader = dyn_synth_prod(seq_len)


    lr = 0.001
    model = DYN_GNN(input_gnn=1, hidden_gnn=hidden_gnn, output_gnn=8, layers_gnn=3, 
                    hidden_rnn=8, sequence_length=seq_len, device = device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 300

    @torch.enable_grad()
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            pred = model(data, data[0].num_graphs)
            y = torch.reshape(data[0].y,(data[0].num_graphs,1))
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)/ data[0].num_graphs
        return total_loss / len(train_loader.dataset[0])

    @torch.no_grad()
    def test(loader):
        model.eval()

        tot_loss = 0
        for data in loader:
            pred = model(data, data[0].num_graphs)
            y = torch.reshape(data[0].y,(data[0].num_graphs,1))          
            loss = F.mse_loss(pred, y)
            tot_loss += loss.item()/ data[0].num_graphs
        return tot_loss/len(loader.dataset[0])

    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}')
        train()     
        train_loss = test(train_loader)
        print(f'Train loss: {train_loss}')
        raw_data.append({'Epoch': epoch, 'Train loss': train_loss, 'Number of timestamps': seq_len, 'GNN hidden dimension': hidden_gnn})

    return raw_data

def main():
    num_nodes = 6
    raw_data = []
    hidden_list = [4,8,16,32]
    for seq_len in range(4,7):
        for hidden_gnn in hidden_list:
            raw_data += model_training(num_nodes, seq_len, hidden_gnn)
    data = pd.DataFrame.from_records(raw_data)
    data.to_csv('dynamics.csv')

if __name__ == '__main__':
    main()