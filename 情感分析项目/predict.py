import re

from LSTMmodel import LSTMModel
from dataSet import loadDataSet
import torch
from flask import Flask, jsonify, request

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

train_loader, test_loader, dict_len, mydict, max_len = loadDataSet()
model = LSTMModel(dict_len, tz_Size=5, input_size=12, hidden_size=12, output_size=2)
model.eval()

save_dict = torch.load('LSTMmodel.pth')
model.load_state_dict(save_dict)


def predict(text_word):
    exp = r'[`,;!。，！]'
    text_word = re.sub(exp, '', text_word)
    text_word = text_word.replace(' ', '')  # 去空
    # 去除特殊字符
    text_word = text_word.strip()

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

        return predict_finall


@app.route('/api/word', methods=['POST'])
def register_user():
    try:
        # 获取请求参数（JSON格式）
        data = request.get_json()
        # 校验必填参数
        if not data:
            return jsonify({"code": 400, "message": "请求体不能为空"}), 400

        # 获取参数
        word = data['word']

        return_back = predict(word)
        return jsonify({"code": 200, "message": "判断成功", "data": return_back}), 200

    except Exception as e:
        return jsonify({"code": 500, "message": f"服务器内部错误: {str(e)}"}), 500


if __name__ == '__main__':
    # while True:
    #     if input("输入q退出程序：") == "q":
    #         break
    #     else:
    #         predict()
    app.run(port=5003)
