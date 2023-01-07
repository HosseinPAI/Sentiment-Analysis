import torch
import torch.nn as nn
import torch.nn.functional as F


class Sentiment140_LSTM(nn.Module):
    def __init__(self, device, input_dim=300, hidden_dim=150, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (
            torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        return hidden


class Sentiment140_Bidirectional_LSTM(nn.Module):
    def __init__(self, device, input_dim=300, hidden_dim=150, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Cause of Bidirectional: 2*hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out, hidden

    def init_hidden(self, batch_size):
        # Cause of Bidirectional: 2*num_of_layer = 2*1 = 2
        hidden = (
            torch.zeros(2, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(2, batch_size, self.hidden_dim).to(self.device))
        return hidden