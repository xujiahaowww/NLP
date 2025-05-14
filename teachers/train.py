import torch
from tqdm import tqdm
from data_deal import *

from model import LSTMModel

device = "cuda" if torch.cuda.is_available() else "cpu"
# 字典大小
dict_size = dict_len

model = LSTMModel(input_size=dict_size)
model.to(device)

# 优化器和损失
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

loss_old = 100 # 用于保存模型
epochs = 10
for epoch in range(epochs):
    pbar = tqdm(train_loader, leave=True, position=0)
    loss_all = 0
    acc_all = 0
    for i, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # 计算模型所需要的参数 lengths
        lengths = torch.sum(x != 0, dim=-1).cpu().long()
        print(lengths)
        out = model(x, lengths)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        loss_avg = loss_all / (i+1)

        # 平均精度
        acc = torch.mean((torch.argmax(out, dim=-1) == y).float())
        acc_all += acc.item()
        acc_avg = acc_all / (i+1)

        # 进度条显示
        pbar.set_description(f"train => epoch:{epoch+1} step:{i+1} loss:{loss_avg:.4f} acc:{acc_avg:.4f}")
        pbar.refresh()

    # 测试过程中不需要更新 禁用梯度计算
    with torch.no_grad():
        pbar = tqdm(test_loader)
        loss_all = 0
        acc_all = 0
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            # 计算模型所需要的参数 lengths
            lengths = torch.sum(x != 0, dim=-1).cpu().long()
            out = model(x, lengths)
            loss = loss_fn(out, y)

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
        torch.save(model.state_dict(), "model.pth")
