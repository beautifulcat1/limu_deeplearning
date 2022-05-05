import random
from xml.etree.ElementTree import TreeBuilder
import torch


#数据的准备

#生成数据
def synthetic_data(w, b, numer_size):
    X = torch.normal(0,1,(numer_size,len(w)))
    y = torch.mm(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y


def data_iter(features, labels, batch_size):
    num_size = len(features)
    indicate = list(range(num_size))
    random.shuffle(indicate)
    for i in range(0, num_size, batch_size):
        batch_indicate = indicate[i:min(i + batch_size,num_size)]
        yield features[batch_indicate], labels[batch_indicate]    
# 线性网络
def liner_net(X,w,b):
    y = torch.mm(X,w) + b
    return y
# 损失函数
def sqrt_loss(y_hat,y):
    loss = 0.5*(y_hat - y)**2
    return loss
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



# 初始化
# 生成数据
true_w = torch.tensor([[1.0],[2.0]])
true_b = 3.0
features, labels = synthetic_data(true_w,true_b,1000)
# 初始化w和b
w = torch.normal(0,0.1,(2,1),requires_grad=True)
b = torch.ones(1, requires_grad=True)
# 初始化线性网络和损失函数
net = liner_net
loss = sqrt_loss
# 初始化批量大小以及训练步长和整个数据训练几次
batch_size = 10
lr = 0.03
epochs = 3

for epoch in range(epochs):
    for X, y in data_iter(features, labels, batch_size):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        los = loss(net(features,w,b),labels)
        print(f"epoch:{epoch+1} loss:{float(los.mean()):f}") 

