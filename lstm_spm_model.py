import torch.nn as nn

class LSTM_SPM(nn.Module):
    def __init__(self):
        super(LSTM_SPM, self).__init__()
        self.input_size = 20
        self.output_size = 1
        self.num_layers = 2
        self.hidden_size = 32
        self.dropout_rate = 0.2

        self.linear_1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear_2 = nn.Linear(self.num_layers * self.hidden_size, self.output_size)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.relu(x)
        
        x = x.unsqueeze(1).repeat(1, 20, 1)

        lstm_out, (h_n, c_n) = self.lstm(x)

        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]
