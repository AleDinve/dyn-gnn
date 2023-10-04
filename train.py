import torch
from torch_geometric import seed
import torch.nn.functional as F
from model import DYN_GNN
import pandas as pd

from wl_test import dyn_wl_generator, dyn_synth_prod

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def model_training(seq_len, hidden_gnn, hidden_rnn, it):
    raw_data = []
    train_loader = dyn_synth_prod(seq_len, batch_size=32)
    # eval_train_loader = dyn_synth_prod(seq_len)



    lr = 0.001
    model = DYN_GNN(input_gnn=1, hidden_gnn=hidden_gnn, output_gnn=hidden_rnn, layers_gnn=6, 
                    hidden_rnn=hidden_rnn, sequence_length=seq_len, device = device)

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
        raw_data.append({'Epoch': epoch, 'Train loss': train_loss, 
                         'Number of timestamps': seq_len, 'GNN hidden dimension': hidden_gnn,
                         'iteration':it})

    return raw_data

def main():
    raw_data = []
    hidden_list = [1,4,8,16,32]
    it = 1
    hidden_rnn = 8
    for it in range(0,10):
        seed.seed_everything(10*(it+1))
        for seq_len in range(4,7):
            print('seq_len: '+str(seq_len))
            for hidden_gnn in hidden_list:
                raw_data += model_training(seq_len, hidden_gnn, hidden_rnn, it)
    data = pd.DataFrame.from_records(raw_data)
    data.to_csv('dynamics_03_10.csv')

if __name__ == '__main__':
    main()