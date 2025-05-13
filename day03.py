import numpy as np
import torch
import torch.nn as nn


# LSTM

class LSTMClass(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClass, self).__init__()

        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> tuple[torch.Tensor]:
        print(x.shape)
        out, (h_last, c_last) = self.LSTM(x)
        print(out.shape)
        out = self.fc(h_last)
        return out, h_last, c_last


if __name__ == '__main__':
    test_data = torch.randn(18, 5, 4)
    model = LSTMClass(4, 5, 8)
    y, h_last, c_last = model(test_data)
    print(y.shape)
    print(h_last.shape, c_last.shape)
