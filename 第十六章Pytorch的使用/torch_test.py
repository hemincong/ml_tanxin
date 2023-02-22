import numpy as np
import torch

# initialize directly from data
data = [[1, 3], [3, 4]]
t_data = torch.tensor(data)
print(f"Initialize directly from data: \n {t_data} \n")
# initlaize from numpy array
np_data = np.array(data)
t_data = torch.from_numpy(np_data)
print(f"Initialize from numpy data: \n {t_data} \n")
# initialize directly from tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"initialize directly from tensor: rand_tensor: \n {rand_tensor} \n")
print(f"Initialize from numpy data: ones_tensor: \n {ones_tensor} \n")
print(f"Initialize from numpy data: zeros_tensor: \n {zeros_tensor} \n")

data = torch.rand(3, 4)
print(f"Shape of data: {data.shape}")
print(f"Datatype of data: {data.dtype}")
print(f"Device data is stored on: {data.device}")

if torch.cuda.is_available():
    tensor = torch.tensor.to('cuda')

print(f"Device data is stored on: {data.device}")

# Tensor也像Numpy
# array支持各种各样的运算操作，比如矩阵乘法、加法、采样等等，而且这些运算均可以在GPU上进行。如果想把
# Tensor在GPU做计算，需要把它先挪到GPU内存中，通过以下几行代码就可以实现:
import torch

# index tensor array
data = torch.ones(4, 4)
data[:, 1] = 0
print(f"Slicing example: \n {data} \n ")
# concatenate 3 tensors
data = torch.rand(3, 3)
t1 = torch.cat([data, data, data], dim=1)
print(f"Concatenation of tensor example before: \n  {data} \n ")
print(f"Concatenation of tensor example after: \n  {t1} \n ")
# multiply tensors
data1 = torch.ones(2, 2)
data2 = torch.ones(2, 2)
mul_res1 = torch.matmul(data1, data2)  # normal multiplication
mul_res2 = data1 * data2  # element-wise multiplication
print(f"normal multiplication example: data1: \n  {mul_res1} \n ")
print(f"normal multiplication example: data2: \n  {mul_res1} \n ")
print(f"normal multiplication example: mul_res1: \n  {mul_res1} \n ")
print(f"element-wise multiplication example: mul_res2 \n  {mul_res2} \n ")

# from torch to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
# change one will change another
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
# from numpy to torch
n = np.ones(5)
t = torch.from_numpy(n)
# change one will change another
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
for itr in range(10):
    prediction = model(data)  # forward pass
    loss = torch.abs(prediction - labels).sum()
    loss.backward()  # backward pass
    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optim.step()  # gradient descent

    print(loss)

import torch

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(6.0, requires_grad=True)
l = 3 * a ** 3 - b ** 2
l.backward()
print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)

# 数据的构造
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

import torch.nn.functional as F
import matplotlib.pyplot as plt


# 模型的构造
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
print(net)

for t in range(50):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print('Accuray = %.2f' % accuracy)
        plt.pause(0.1)
plt.show()
