# nn.GRUCell 类似 nn.RNNCell 分类任务 多对多输出 类别是hize两倍 多对一的输出 类别是hize两倍
# nn.GRU 类似 nn.RNN nn.LSTM 分类任务 多对多输出 类别是hize两倍 多对一的输出 类别是hize两倍
#
# 多层 双向 在这个基础上输出分类任务
# 我做之前的数据集做的145词表大小的分类

import torch
import torch.nn as nn
from torch.nn import GRUCell, GRU


# 数据准备
def data_P(text):
    text_value = ' '.join(i for i in text).split()
    print(text_value)
    text_value = set(text_value)

    dic_word2id = {value: key for key, value in enumerate(text_value)}
    dic_id2word = {key: value for key, value in enumerate(text_value)}

    test_list_x = []
    test_list_y = []
    for word in text:
        a = [dic_word2id[i] for i in word.split()[:-1]]
        b = dic_word2id[word.split()[-1]]
        test_list_x.append(a)
        test_list_y.append(b)

    return test_list_x, test_list_y, len(text_value)


class GRU_Net(nn.Module):
    def __init__(self, c_number, input_size, hidden_size, output_size, if_gru=True):
        super(GRU_Net, self).__init__()
        self.hidden_size = hidden_size
        self.if_gru = if_gru
        self.embedding = nn.Embedding(num_embeddings=c_number, embedding_dim=input_size)
        self.gru = GRU_Net.judge(self, input_size, hidden_size)
        self.grucell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def judge(self, *kwarg):
        input_size, hidden_size = kwarg
        if self.if_gru:
            fn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        else:
            # 创建GRUCell方法
            def fn_1(x):
                batch_size, seq_len, dim = x.size(0), x.size(1), x.size(2)
                # 数据处理为（S,B,D）
                x = x.permute(1, 0, 2)

                x_back = torch.flip(x, dims=[1])
                x_back = x_back.permute(1, 0, 2)
                seq_len_back = x_back.size(1)

                # 正向循环遍历每个时间步
                hiddens = []
                x_last = None
                for t in range(seq_len):
                    # 输入数据
                    x_t = x[t]
                    hidden_t = self.grucell(x_t)
                    hiddens.append(hidden_t)
                    x_last = hidden_t
                # 所有时间步堆叠成新的张量
                output = torch.stack(hiddens)
                # 对x_last进行升维
                x_last = torch.unsqueeze(x_last, 0)

                # 反向循环遍历每个时间步
                hiddens_back = []
                x_last_back = None
                for t in range(seq_len_back):
                    # 输入数据
                    x_t = x[t]
                    hidden_t = self.grucell(x_t)
                    hiddens_back.append(hidden_t)
                    x_last_back = hidden_t
                # 所有时间步堆叠成新的张量
                output_back = torch.stack(hiddens_back)
                # 对x_last进行升维
                x_last_back = torch.unsqueeze(x_last_back, 0)

                # 处理数据
                output = output.permute(1, 0, 2)
                output_back = output_back.permute(1, 0, 2)
                output_finall = torch.cat((output, output_back), dim=2)

                x_finall = torch.cat((x_last, x_last_back), dim=0)
                x_finall = torch.mean(x_finall, dim=0).unsqueeze(0)
                return output_finall, x_finall

            fn = fn_1

        return fn

    def forward(self, x):
        x = self.embedding(x)
        h_all, h_last = self.gru(x)
        print(h_all.shape, h_last.shape)

        # 多对一
        h_last = torch.mean(h_last, dim=0).unsqueeze(0)
        out = self.fc(h_last)
        # 多对多
        # out = self.fc(h_all)
        return out


class GRUCell_Net(GRU_Net):
    def __init__(self, c_number, input_size, hidden_size, output_size, if_gru=False):
        super(GRUCell_Net, self).__init__(
            c_number,
            input_size,
            hidden_size,
            output_size,
            if_gru
        )

    def forward(self, x):
        x = self.embedding(x)
        h_all, h_last = self.gru(x)
        # h_last = torch.mean(h_last, dim=0)
        out = self.fc(h_last)
        return out


if __name__ == '__main__':
    text = ["she bakes cakes", "we watch movies", "they build forts", "he fixes bikes", "you drink tea",
            "it spins fast", "dogs chase tails", "birds build nests", "fish hide well", "kids build sandcastles",
            "sun sets late", "moon rises early", "stars fade slowly", "wind howls loud", "rain pours heavy",
            "leaves turn brown", "grass grows green", "clouds drift lazily", "waves crash hard",
            "volcanoes erupt violently",
            "robots assemble cars", "computers process data", "phones ring loud", "radios play music",
            "cameras capture moments",
            "writers craft stories", "painters mix colors", "dancers spin gracefully", "actors memorize lines",
            "singers hit notes",
            "chefs chop veggies", "bakers knead dough", "farmers plant seeds", "doctors heal wounds",
            "teachers grade papers",
            "police catch thieves", "firefighters save lives", "soldiers defend borders", "pilots fly jets",
            "sailors navigate seas",
            "astronauts float weightlessly", "scientists test theories", "engineers design bridges",
            "architects sketch plans", "programmers debug code",
            "babies crawl slowly", "toddlers explore eagerly", "teens text constantly", "adults work hard",
            "elders reminisce fondly"]
    test_list_x, test_list_y, cb_len = data_P(text)
    test_list_x = torch.LongTensor(test_list_x)
    model = GRU_Net(cb_len, input_size=5, hidden_size=6, output_size=cb_len)
    finall = model(test_list_x)
    print(finall.shape)

    model2 = GRUCell_Net(cb_len, input_size=5, hidden_size=6, output_size=cb_len)
    finall2 = model2(test_list_x)
    print(finall2.shape)
