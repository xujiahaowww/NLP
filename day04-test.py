import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        # 不写embedding了简化数据
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.avgPOOL = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (l_last, c_last) = self.lstm(x)

        print(out.shape)
        print(l_last.shape)
        out = self.fc(l_last)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    test = torch.randn((2, 5, 8))
    model = LSTM(8, 10, 5)

    test_out = model(test)
    print(test_out.shape)
