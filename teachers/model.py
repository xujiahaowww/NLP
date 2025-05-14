import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def  __init__(self, input_size=512, hidden_size=512, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        # 词嵌入
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 多层感知机mlp
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            # 丢弃 为了防止过拟合
            nn.Dropout(0.2),
        )
        # 第一个分支
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,dropout=0.2)
        self.avg1 = nn.AdaptiveAvgPool1d(1)

        #  第二个分支
        self.avg2 = nn.AdaptiveAvgPool1d(1)

        # 输出层
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x,lengths):
        """
        :param x: 二维数据 【批次，词数】
        :return: 【批次，分类类别数】
        """
        embed = self.embedding(x)
        linear1 = self.mlp1(embed)

        # 第一个分支
        # 打包 目的取非填充部分数据 减少计算量
        # 将填充序列打包
        packed_embedded = pack_padded_sequence(linear1, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_embedded)
        # 打包 还原成原本的数据
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]
        avg1 = self.avg1(lstm_out.permute(0, 2, 1)).squeeze(-1)

        #  第二个分支
        avg2 = self.avg2(linear1.permute(0, 2, 1)).squeeze(-1)

        # 融合
        out = torch.cat((avg1, avg2), dim=-1)

        # 输出层
        output = self.mlp2(out)

        return output

# 测试
if __name__ == '__main__':
    model = LSTMModel()
    x = torch.randint(0, 256, (64, 256))
    lengths = torch.randint(1, 256, (64,))
    out = model(x, lengths)
    print(out.shape)