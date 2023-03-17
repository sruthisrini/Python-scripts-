import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F

class RNNDataset(Dataset):
    def __init__(self, in_data, in_labels):
        self.data = in_data
        self.labels = in_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, n_class, device,
                 lstm_n_layers=2,
                 lstm_hidden_dim=32,
                 bidirect=True,
                 add_feat=False,
                 dropout_rate=0.2,
                 activation='sigmoid',
                 add_fc_layer=True,
                 embed=True,
                 embed_vocab_size=5,
                 embed_dim=10):

        super(LSTMModel, self).__init__()

        self.bidirect = bidirect
        self.embed = embed
        self.add_feat = add_feat
        self.lstm_n_layers = lstm_n_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        self.add_fc_layer = add_fc_layer
        self.embedding = nn.Embedding(embed_vocab_size, embed_dim)
        # Dropout.
        self.dropout = nn.Dropout(dropout_rate)
        # LSTM.
        if embed:
            if add_feat:
                self.lstm = nn.LSTM(embed_dim + (input_dim - 1), lstm_hidden_dim, lstm_n_layers,
                                   bidirectional=bidirect, bias=True,
                                   batch_first=True).to(device)
            else:
                self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, lstm_n_layers,
                                   bidirectional=bidirect, bias=True,
                                   batch_first=True).to(device)
        else:
            self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_n_layers,
                               bidirectional=bidirect, bias=True,
                               batch_first=True).to(device)
        if bidirect:
            if add_fc_layer:
                self.fc1 = nn.Linear(2*lstm_hidden_dim, lstm_hidden_dim)
                self.fc2 = nn.Linear(lstm_hidden_dim, n_class)
            else:
                self.fc = nn.Linear(2*lstm_hidden_dim, n_class)
        else:
            if add_fc_layer:
                self.fc1 = nn.Linear(lstm_hidden_dim, int(lstm_hidden_dim/2))
                self.fc2 = nn.Linear(int(lstm_hidden_dim/2), n_class)
            else:
                self.fc = nn.Linear(lstm_hidden_dim, n_class)


    def zero_state(self, batch_size):
        if self.bidirect:
            h0 = torch.zeros(2*self.lstm_n_layers, batch_size, self.lstm_hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
            c0 = torch.zeros(2*self.lstm_n_layers, batch_size, self.lstm_hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
        else:
            h0 = torch.zeros(self.lstm_n_layers, batch_size, self.lstm_hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
            c0 = torch.zeros(self.lstm_n_layers, batch_size, self.lstm_hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
        return h0, c0

    def forward(self, batch_data, batch_lens, batch_size):
        h0, c0 = self.zero_state(batch_size)

        if self.embed:
            x_embed = self.embedding(batch_data[:, :, 0].long()).clone().detach().requires_grad_(True)
            if self.add_feat:
                x_embed = torch.cat([x_embed, batch_data[:, :, 1:]], dim=2)
        else:
            x_embed = batch_data.clone().detach().requires_grad_(True)

        x_packed = pack_padded_sequence(x_embed, batch_lens, batch_first=True, enforce_sorted=False)

        if self.bidirect:
            out, (hidden, cell) = self.lstm(x_packed, (h0, c0))
            hidden = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)
        else:
            out, (hidden, cell) = self.lstm(x_packed, (h0, c0))

        x = self.dropout(hidden)

        if self.add_fc_layer:
            x = self.dropout(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc(x)

        return x, x_embed


if __name__ == "__main__":
    
    test_model = LSTMModel(input_dim=4, n_class=30,activation='sigmoid', device="cpu")   # for multi-label
    test_model = LSTMModel(input_dim=4, n_class=1,activation='sigmoid', device="cpu")    # for binary
     
