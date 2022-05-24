"""
该模型为单一变量预测功率，数据集规模较小，故未使用dataset和dataloader，
"""


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional
import torch.backends.cudnn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"NOTE: Set random seed: {seed}. \n")

set_seed(122) #设置随机数种子

class LstmRNN(nn.Module): # 该模型来自论坛,相关语法查看：python类与对象
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1,num_layers = 1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.forwardCalculation = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


datas = pd.read_csv("D:\station00.csv")  # 引用数据
lmd = datas.values[:970,8]  #取前十天
power = datas.values[:970,14]



# 划分训练集，测试集（未打乱顺序）
data_len = 970
train_data_ratio = 0.8
train_data_len = int(data_len*train_data_ratio) #776
train_lmd,test_lmd = lmd[:train_data_len].astype("float32"),lmd[train_data_len:].astype("float32")
train_power,test_power = power[:train_data_len].astype("float32"),power[train_data_len:].astype("float32")



#此时x.y均为一维array，需要转化为三维放入模型，以下使用reshape方法转化
INPUT_FEATURES_NUM = 1
OUTPUT_FEATURES_NUM = 1
train_power = train_power.reshape(-1,776, INPUT_FEATURES_NUM)
print(type(train_power)) # 内置函数type()查看数据类型
train_lmd = train_lmd.reshape(-1,776, INPUT_FEATURES_NUM)
#转化数据类型为Tensor
train_power_tensor = torch.from_numpy(train_power)
print(type(train_power_tensor))
train_lmd_tensor = torch.from_numpy(train_lmd)




#同上对测试集进行转化
test_power = test_power.reshape(-1,194, INPUT_FEATURES_NUM)
test_lmd = test_lmd.reshape(-1,194, INPUT_FEATURES_NUM)
test_power_tensor = torch.from_numpy(test_power)
test_lmd_tensor = torch.from_numpy(test_lmd)

#实例化对象（lstm模型）
lstm_model = LstmRNN()
print('LSTM model:', lstm_model)
print('model.parameters:', lstm_model.parameters)

loss_function = nn.MSELoss()  #损失函数MSE
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2) # 优化器Adam，学习率0.01


#训练
epoch = 1000
for i in range(epoch):
    output = lstm_model(train_lmd_tensor) #数据输入lstm
    loss = loss_function(output, train_power_tensor) #计算损失
    loss.backward()  #反馈
    optimizer.step() #更新梯度
    optimizer.zero_grad()  #梯度清零
    if (i+1) % 20 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(i+1, epoch, loss.item()))
        print("The loss value is reached")


predictive_y_for_training = lstm_model(train_lmd_tensor )
predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy() #tensor转回array（画图用）

lstm_model = lstm_model.eval() # 此处对model使用eval方法，个人理解：防止测试集数据改变权重（ctrl+左键查看源码注释）

predictive_y_for_testing = lstm_model(test_lmd_tensor)
predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()




# 可视化
'''
plt用法较简单，可查看官方文档，不再逐行注释
'''
x_test = np.arange(776,970)
x_train = np.arange(0,776)
plt.figure(figsize=(30, 7))
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(power) #折线1
plt.plot(x_train, predictive_y_for_training) #折线2
plt.plot(x_test, predictive_y_for_testing)  #折线3
plt.xlabel('time')
plt.ylabel('power')
plt.show()
