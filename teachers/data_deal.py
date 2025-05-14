"""
    数据处理部分 思考：需要哪些步骤
    数据清洗
    分词 可做可不做
    重点：构建词表 字典
    重点：根据字典转成编码形式 文本转索引 得到【批次，词数】格式
    重点：数据填充 统一数据大小
    重点：（划分数据集） 批量加载器
    模型：词嵌入
"""
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

data_path = "./hotel_discuss2.csv"
dict_file = "data/dict.txt"
encoding_file = "data/encoding.txt"
# 筛选不需要的符号
filter_symbol = "。"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据处理
def dict_data():
    mydict = {}
    # 添加填充符号
    mydict["<pad>"] = 0
    code = 1
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            each_line = line.strip() # 换行符 空格等去掉

            for word in each_line:
                # 筛选不需要的符号
                if word in filter_symbol:
                    continue
                # 每个词放入字典
                if word not in mydict:
                    mydict[word] = code
                    code += 1
        # 存放一些未知字符 空字符
        mydict["<unk>"] = code

    return mydict

mydict = dict_data()

# 字典存放到文件
# with open(dict_file, "w", encoding="utf-8-sig") as f:
#     f.write(str(mydict))
#     print("字典保存成功")

# 文本转索引
with open(data_path, "r", encoding="utf-8-sig") as f:
    with open(encoding_file, "w", encoding="utf-8-sig") as fw:
        for line in f.readlines():
            each_line = line.strip()
            label = each_line[0] # 标签
            content = each_line[2:-1] # 内容

            for ch in content:
                if ch in filter_symbol:
                    continue
                else:
                    fw.write(str(mydict[ch]) + ",")

            fw.write("\t" + str(mydict[label]) + "\n")
print("文本转索引成功")

# 字典长度
dict_len = len(mydict)

# 填充统一大小 方法一：设定一个最大值 方法二：找到最长的话 以最长句子大小来填充
values = []
labels = []
max_lens = 256
with open(encoding_file, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()
    # random.shuffle(lines) # 随机打乱数据

    for line in lines:
        content,label = line.strip().split("\t")
        content = [int(i) for i in content.split(",") if i.isdigit()]

        # 作用是每行数据进行填充 填充到设定的最大值
        for i in range(max_lens - len(content)):
            content.append(0)

        content = content[:max_lens]

        values.append(content)
        labels.append(int(label))

# 列表转数组
values = np.array(values, dtype="int64")
labels = np.array(labels, dtype="int64")
print(labels)
print(values.shape, labels.shape)

# 划分数据集
x_train, x_test, y_train, y_test= train_test_split(values,  labels, test_size=0.2,random_state=42)

# 转换为张量 小批量加载 from_numpy 同一个内容 就地 tensor 会复制数组
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

if __name__ == "__main__":
    for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        print(x, y)
        break




