import random
import torch

# 生成带标签的数据
def sythetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = sythetic_data(true_w,true_b,1000)

# 生成小批量数据作为样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)#样本的大小
    indices = list(range(num_examples))#样本的下标
    random.shuffle(indices)#将下标打乱

    #每次产出一个批量，并记住位置，下次从这个位置继续产出一个批量
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
  #向量[]用法     向量[0,0],直接取第0行0列的值， 向量[[0],[0]],将第0行0列作为值作为一个值，  向量1[向量2]，取下标为向量2各分量位置的向量1的向量如 a[[1,2,3,4,10]],在a向量中取位置为1,2,3,4,10的向量

# 线性神经网络
def linreg(X,w,b):
    return torch.mm(X,w) + b
# 平方损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2
# 梯度下降更新
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练过程

# 参数的初始化
w = torch.normal(0,0.01, (2,1), requires_grad=True)
b = torch.ones(1, requires_grad=True)

# 超参数的初始化
batch_size = 10
lr = 0.03
num_epochs = 3

# 训练的网络和损失函数的选择
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss (net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss (net(features,w,b),labels)
        print(f"epoch {epoch + 1}, loss{float(train_l.mean()):f}")