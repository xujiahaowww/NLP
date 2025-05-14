import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, cb_size, tz_Size, input_size=12, hidden_size=12, output_size=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=cb_size, embedding_dim=tz_Size)

        self.feature_1 = nn.Sequential(
            nn.LayerNorm(tz_Size),
            nn.Linear(tz_Size, input_size),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 序列化1
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        # 序列化2
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        self.feature_2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, lengths):
        x = self.embedding(x)  # 转成3维结构
        x = self.feature_1(x)

        # 打包 目的取非填充部分数据 减少计算量
        # 将填充序列打包
        packed_embedded = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x_1, (h_last, c_last) = self.LSTM(packed_embedded)
        # 打包 还原成原本的数据
        x_1 = pad_packed_sequence(x_1, batch_first=True)[0]

        x_1 = x_1.permute(0, 2, 1)
        x_1 = self.avgpool_1(x_1).squeeze(dim=-1)  # 得到句向量

        x_2 = x.permute(0, 2, 1)
        x_2 = self.avgpool_2(x_2).squeeze(dim=-1)  # 得到句向量

        x_cat = torch.cat((x_1, x_2), dim=-1)  # 残差操作
        x_cat = self.feature_2(x_cat)
        return x_cat


if __name__ == '__main__':
    testdata = torch.LongTensor(np.random.randint(low=0, high=255, size=[100, 10]))
    lengths = torch.randint(1, 2, (100,))
    print(testdata.shape)
    model = LSTMModel(cb_size=2000, tz_Size=5, input_size=12, hidden_size=12, output_size=2)
    output = model(testdata,lengths)
    print(output.shape)
