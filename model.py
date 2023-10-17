from torch_geometric.nn import GraphConv, global_add_pool, GINConv, BatchNorm
import torch
from torch.nn import RNN, Linear, Sequential,  BatchNorm1d, ReLU, Tanh
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

class DYN_GNN(torch.nn.Module):
    def __init__(self, input_gnn, hidden_gnn, output_gnn, layers_gnn, 
                 hidden_rnn, sequence_length, device = device, conv_type= 'gin'):
        super().__init__()
        self.input_gnn = input_gnn
        self.hidden_gnn = hidden_gnn
        self.output_gnn = output_gnn # per ora
        self.seq_len = sequence_length
        self.gnns = [] 
        for _ in range(sequence_length):
            self.gnns.append(torch.nn.ModuleList())
            input_gnn = self.input_gnn
            for _ in range(layers_gnn):
                if conv_type == 'gconv':
                    self.gnns[-1].append(GraphConv(input_gnn, hidden_gnn, aggr='add', bias=True).to(device))
                else:
                    self.gnns[-1].append(GINConv(Sequential(Linear(input_gnn, hidden_gnn),
                       BatchNorm1d(hidden_gnn),
                        Tanh(),
                       Linear(hidden_gnn, hidden_gnn), Tanh())).to(device))
                input_gnn = hidden_gnn

        
        self.linear_gnn = Linear(hidden_gnn, self.output_gnn).to(device)
        self.rnn = RNN(self.output_gnn, hidden_rnn).to(device)
        self.linear_rnn = Linear(hidden_rnn,1).to(device)

    def forward(self, data, batch_size):
        # x: ordered list of torch_geometric.data 
        h = torch.zeros(self.seq_len, batch_size, self.output_gnn).to(device)
        batch = data[0].batch.to(device)
        for count, gnn in enumerate(self.gnns):

            single_data = data[count].to(device)
            x = single_data.x
            #print(x.is_cuda)
            for conv in gnn:   
                #print([param.is_cuda for param in conv.parameters()])             
                x = conv(x, single_data.edge_index)
            x = global_add_pool(x, batch)
            x = self.linear_gnn(x)

            h[count, :, :] = torch.reshape(x, (-1, self.output_gnn))
        y = self.rnn(h)
        return torch.sigmoid(self.linear_rnn(torch.squeeze(y[-1],dim=0)))
    
