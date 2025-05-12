import torch
import torch.nn as nn
import jieba


def test01():
    text2 = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    value2 = jieba.lcut(text2)

    dic1 = {}
    value2 = list(set(value2))
    for index, i in enumerate(value2):
        dic1[i] = index

    print(dic1)
    # embedding_dim一般来说2的n方
    chuli = nn.Embedding(num_embeddings=len(value2), embedding_dim=4)
    for i in value2:
        index = dic1[i]
        word_embd = chuli(torch.LongTensor([index]))
        print(i, word_embd)


# 数据准备方法
def data_P(text):
    text_value = [' '.join(jieba.lcut(i)) for i in text]
    arr = []
    for i in text_value:
        a = i.split()
        arr += a
    text_arr = list(set(arr))
    dic_word = {key: i for i, key in enumerate(text_arr)}
    dic_index = {i: key for i, key in enumerate(text_arr)}
    simple_arr = []
    target_arr = []
    for eee in text:
        input = [dic_word[i] for i in eee.split()[:-1]]
        target = dic_word[eee.split()[-1]]
        simple_arr.append(input)
        target_arr.append(target)

    return simple_arr, target_arr, text_arr, dic_word, dic_index


# NNLM模型初次搭建
class NNLM_Model(nn.Module):
    def __init__(self, text_arr, tz_size):
        super(NNLM_Model, self).__init__()
        self.emb = nn.Embedding(num_embeddings=len(text_arr), embedding_dim=tz_size)
        self.fc1 = nn.Linear(in_features=2 * tz_size, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=len(text_arr))

    def forward(self, x):
        print(x.shape)
        x = self.emb(x)  # [4,2,5]

        x = x.view(x.size(0), -1)  # [4,10]
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    text_train = [
        "she bakes cakes", "we watch movies", "they build forts", "he fixes bikes", "you drink tea",
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
        "elders reminisce fondly"
    ]
    # 数据准备
    # 训练集
    simple_arr, target_arr, text_arr, dic_word, dic_index = data_P(text_train)

    simple_arr = torch.tensor(simple_arr)
    target_arr = torch.tensor(target_arr)

    # 测试集
    text_test = ["she love cakes", "babies crawl slowly", "toddlers explore eagerly"]
    text_test_arr = []
    print('cakes' in dic_word)
    for aaa in text_test:
        a = [(dic_word[i] if i in dic_word else 10) for i in aaa.split()[:-1]]
        text_test_arr.append(a)

    text_test_arr = torch.tensor(text_test_arr)

    model = NNLM_Model(text_arr, tz_size=5)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(simple_arr.shape)
    for epoch in range(3000):
        y_pred = model(simple_arr)
        loss = lossFn(y_pred, target_arr)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # 测试
    _, predict = torch.max(model(text_test_arr), 1)
    print(predict)

    predict = predict.numpy()
    predict = [dic_index[i] for i in list(predict)]

    print(text_test)
    print(predict)