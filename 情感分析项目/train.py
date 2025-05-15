import torch
from torch import optim
from dataSet import loadDataSet
from LSTMmodel import LSTMModel
import torch.nn as nn
from tqdm import tqdm

train_loader, test_loader, dict_len, _, _ = loadDataSet()

print('词表字典长度', dict_len)

epochs = 10
model = LSTMModel(dict_len, tz_Size=5, input_size=12, hidden_size=12, output_size=2)

lr = 0.1
optimizer = optim.Adam(model.parameters(), lr)
lossFn = nn.CrossEntropyLoss()

model.train()


def adjust_learning_rate(optimizer, epoch, start_lr):
    # 每三个epoch衰减一次
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(epochs):
    lossALL = 0
    successAll = 0

    adjust_learning_rate(optimizer, epoch, lr)
    # 查看每一轮的学习率情况
    print("Epoch:{}  Lr:{:.2E}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    pbar = tqdm(train_loader, leave=True, position=0)
    for i, (x, y) in enumerate(pbar):
        lengths = torch.sum(x != 0, dim=-1).long()
        pre_y = model(x, lengths)
        loss = lossFn(pre_y, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lossALL += loss.item()

        _, predicted = torch.max(pre_y.data, 1)

        successAll += (predicted == y).sum().item()  # 求得一批次里多少个正确
        lossAvg = lossALL / (i + 1)
        successAvg = successAll / ((i + 1) * 64)
        # 进度条显示
        pbar.set_description(f"train => 第:{epoch + 1}轮 批次:{i + 1} loss:{lossAvg:.4f} success:{successAvg:.4f}")
        pbar.refresh()

    # # 测试过程中不需要更新 禁用梯度计算
    with torch.no_grad():
        pbar = tqdm(test_loader)
        loss_all = 0
        acc_all = 0
        loss_old = 10
        for i, (x, y) in enumerate(pbar):
            # 计算模型所需要的参数 lengths
            lengths = torch.sum(x != 0, dim=-1).cpu().long()
            out = model(x, lengths)
            loss = lossFn(out, y)
            loss_all += loss.item()
            loss_avg = loss_all / (i + 1)

            # 平均精度
            acc = torch.mean((torch.argmax(out, dim=-1) == y).float())
            acc_all += acc.item()
            acc_avg = acc_all / (i + 1)

            # 进度条显示
            pbar.set_description(f"test => epoch:{epoch + 1} step:{i + 1} loss:{loss_avg:.4f} acc:{acc_avg:.4f}")
            pbar.refresh()

        if loss_avg < loss_old:
            loss_old = loss_avg
            torch.save(model.state_dict(), "LSTMmodel.pth")
