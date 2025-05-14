import re
from LSTMmodel import LSTMModel
from dataSet import loadDataSet
import torch

train_loader, test_loader, dict_len, mydict, max_len = loadDataSet()
model = LSTMModel(dict_len, tz_Size=5, input_size=12, hidden_size=12, output_size=2)
model.eval()

save_dict = torch.load('LSTMmodel.pth')
model.load_state_dict(save_dict)

text_word = '服务挺不错'

exp = r'[`,;!。，！]'
text_word = re.sub(exp, '', text_word)
text_word = text_word.replace(' ', '')  # 去空

text_word_list = [i for i in text_word]

text_id_list = [mydict[i] for i in text_word_list]
print(text_id_list)
# 填充预测数据
for i in range(max_len - len(text_id_list)):
    text_id_list.append(0)
text_id_list = torch.LongTensor(text_id_list).unsqueeze(0)
lengths = torch.sum(text_id_list != 0, dim=-1).long()

with torch.no_grad():
    predict_finall = None
    output = model(text_id_list, lengths)
    _, predicted = torch.max(output.data, 1)
    # 二分类判断
    if predicted.item() == 1:
        predict_finall = '好评'
    else:
        predict_finall = '差评'
    print(predict_finall)
