import torch
import torch.nn as nn


# Definición de la arquitectura de la red neuronal
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers=2):
        super(TimeSeriesPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = lstm_out[:, -1, :]
        output = self.linear(lstm_out)

        return output


# Definición de la arquitectura de la red neuronal para Modelo 2
class TimeSeriesPredictor_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers=3):
        super(TimeSeriesPredictor_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = lstm_out[:, -1, :]
        output = self.linear(lstm_out)

        return output
    

class TimeSeriesPredictor_ModelWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length,
                 num_layers=2, dropout_prob=0.2):
        super(TimeSeriesPredictor_ModelWithDropout, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0),
                          self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)

        return output
    
