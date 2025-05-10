import torch
import torch.nn as nn
import jieba


# text = ['北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。', '还有不少外国运动员在玩']
# value = [' '.join(jieba.cut(i)) for i in text]
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


def data_P(text):
    text_value = [' '.join(jieba.lcut(i)) for i in text]
    arr = []
    for i in text_value:
        a = i.split()
        arr += a

    text_arr = list(set(arr))

    dic_word = {key: i for i, key in enumerate(text_arr)}
    dic_index = {i: key for i, key in enumerate(text_arr)}

    #  text = ["i like dog", "i love coffee", "i hate milk", "i do nlp"]
    simple_arr = []
    target_arr = []
    for eee in text:
        input = [dic_word[i] for i in eee.split()[:-1]]
        target = dic_word[eee.split()[-1]]
        simple_arr.append(input)
        target_arr.append(target)

    return simple_arr, target_arr, text_arr, dic_word, dic_index


# NNLM模型初次搭建
class NNLModel(nn.Module):
    def __init__(self, text_arr, tz_size):
        super(NNLModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=len(text_arr), embedding_dim=tz_size)
        self.fc1 = nn.Linear(in_features=2 * tz_size, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=len(text_arr))

    def forward(self, x):
        x = self.emb(x)  # [4,2,5]
        x = x.view(x.size(0), -1)  # [4,10]
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    text = ["i like dog", "i love coffee", "i hate milk", "i do nlp"]

    # 数据准备
    simple_arr, target_arr, text_arr, dic_word, dic_index = data_P(text)

    simple_arr = torch.tensor(simple_arr)
    target_arr = torch.tensor(target_arr)

    model = NNLModel(text_arr, tz_size=5)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(400):
        y_pred = model(simple_arr)
        loss = lossFn(y_pred, target_arr)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    _, predict = torch.max(model(simple_arr), 1)
    print(predict)

    predict = predict.numpy()
    predict = [dic_index[i] for i in list(predict)]

    print(text)
    print(predict)
