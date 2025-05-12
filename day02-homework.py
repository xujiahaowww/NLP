import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

text = "Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers."

text_list = text.split(" ")
print(text_list)

# 定义WORD2VEC模型
model = Word2Vec([text_list], vector_size=2, window=1, min_count=1, sg=0)

# 获取词汇表
word_list = list(model.wv.index_to_key)
list = [model.wv[word] for word in word_list]
shape = (len(list), len(list[0]))
print(shape)

# 词向量转句向量
list_np = np.array(list)
sentence_list = list_np.mean(axis=0)
print(sentence_list)

a = torch.randn(2, 3, 5)

