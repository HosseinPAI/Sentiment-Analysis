import torch
import torch.nn as nn
import torch.nn.functional as F


class Sentiment140_Pyramid_LSTM(nn.Module):
    def __init__(self, device, input_dim=300, hidden_dim=150, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(2 * hidden_dim, 2 * hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(4 * hidden_dim, 4 * hidden_dim, batch_first=True)
        self.lstm4 = nn.LSTM(8 * hidden_dim, 8 * hidden_dim, batch_first=True)
        self.fc = nn.Linear(8 * hidden_dim, output_dim)

    def forward(self, x, hidden):
        # input Shape = [batch_size*280*300]
        out1, hidden[0] = self.lstm1(x, hidden[0])  # out1 Shape = [batch_size*280*64]
        out1_cat = torch.cat((out1[:, ::2, :], out1[:, 1::2, :]),
                             dim=2)  # Concatenate Out1 (2i,2i+1) ->  out1_cat Shape = [batch_size*140*128]
        out2, hidden[1] = self.lstm2(out1_cat, hidden[1])  # out2 Shape = [batch_size*140*128]
        out2_cat = torch.cat((out2[:, ::2, :], out2[:, 1::2, :]),
                             dim=2)  # Concatenate Out2 (2i,2i+1) ->  out2_cat Shape = [batch_size*70*256]
        out3, hidden[2] = self.lstm3(out2_cat, hidden[2])  # out3 Shape = [batch_size*70*256]
        out3_cat = torch.cat((out3[:, ::2, :], out3[:, 1::2, :]),
                             dim=2)  # Concatenate Out3 (2i,2i+1) ->  out3_cat Shape = [batch_size*35*512]
        out4, hidden[3] = self.lstm4(out3_cat, hidden[3])  # out4 Shape = [batch_size*35*512]

        out = out4[:, -1, :]  # Select the Last output for Clasification
        out = self.fc(out)  # out Shape : [batch_size*3]
        out = F.softmax(out, dim=1)  # out Shape : [batch_size*3]
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = [(torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                   torch.zeros(1, batch_size, self.hidden_dim).to(self.device)),
                  (torch.zeros(1, batch_size, self.hidden_dim * 2).to(self.device),
                   torch.zeros(1, batch_size, self.hidden_dim * 2).to(self.device)),
                  (torch.zeros(1, batch_size, self.hidden_dim * 4).to(self.device),
                   torch.zeros(1, batch_size, self.hidden_dim * 4).to(self.device)),
                  (torch.zeros(1, batch_size, self.hidden_dim * 8).to(self.device),
                   torch.zeros(1, batch_size, self.hidden_dim * 8).to(self.device))]
        return hidden
