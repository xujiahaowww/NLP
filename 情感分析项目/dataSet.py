from collections import OrderedDict

import numpy as np
import pandas as pd
import csv
import torch
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def loadDataSet():
    text_path = './hotel_discuss2.csv'
    exp = r'[`,;!。，！]'

    mydict = {}

    with open(text_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 得到词表字典
        target_list = []
        dict_list_all = []
        liable_list_all = []
        for row in reader:
            comment_id = row[0]
            comment_text = row[1]
            comment_text = re.sub(exp, '', comment_text)  # 去掉部分特殊符号
            comment_text = comment_text.replace(' ', '')  # 去空
            dict_list = [i for i in comment_text]

            liable_list_all.append(re.sub(r'\ufeff', '', comment_id))
            dict_list_all.append(dict_list)
            for word in comment_text:
                target_list.append(word)
        target_list = list(OrderedDict.fromkeys(target_list))  # 去重防止序列错位
        mydict = {key: v + 1 for v, key in enumerate(target_list)}
        # 填充字符的序号
        mydict["<pad>"] = 0
        mydict["unknown"] = len(mydict)
        # liable_list_all 已处理的目标集 ， mydict字典k-v  ,
        finall_list = []
        for centence in dict_list_all:
            list1 = []
            for word in centence:
                if word in mydict:
                    list1.append(mydict[word])
                else:
                    list1.append(mydict["unknown"])
            finall_list.append(list1)

        # 有了词和目标，对词进行填充
        max_list = max((x for x in finall_list if isinstance(x, list)), key=len, default=None)
        max_len = len(max_list)
        print(max_len)

        tc_finall_list = []
        for sentence in finall_list:
            for i in range(max_len - len(sentence)):
                sentence.append(0)

            tc_finall_list.append(sentence)

        # 列表转数组
        values = np.array(tc_finall_list, dtype="int64")
        labels = np.array(liable_list_all, dtype="int64")
        print(labels)
        print(values.shape, labels.shape)

        # 划分数据集
        x_train, x_test, y_train, y_test = train_test_split(values, labels, test_size=0.2, random_state=27)

        # 转换为张量 小批量加载 from_numpy 同一个内容 就地 tensor 会复制数组
        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

        dict_len = len(mydict)

        return train_loader, test_loader, dict_len, mydict, max_len
