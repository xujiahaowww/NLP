import torch
import torch.nn as nn
from gensim.models import Word2Vec


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


class RNN(nn.Module):
    def __init__(self, cb_len, tz_size, hidden_size=8):
        super(RNN, self).__init__()
        # 第一步对输入的数据进行矩阵映射，词嵌入获取词向量维度
        self.emb = nn.Embedding(num_embeddings=cb_len, embedding_dim=tz_size)
        # 第二步进行RNN循环
        self.rnn = nn.RNN(input_size=tz_size, hidden_size=hidden_size, batch_first=True,)

        self.fc = nn.Linear(in_features=hidden_size, out_features=cb_len)

    def forward(self, x):
        x = self.emb(x)
        out, h_last = self.rnn(x)
        print(out.shape)
        print(h_last.shape)
        out = self.fc(out)
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

    model = RNN(cb_len, tz_size=5)
    test_list_x = torch.LongTensor(test_list_x)
    pre_y = model(test_list_x)
    print('最终输出y', pre_y.shape)
