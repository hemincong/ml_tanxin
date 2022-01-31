# 引用 sklearn库,主要为了使用其中的线性回归模块
from sklearn import datasets, linear_model
# train_test_split用来把数据集拆分为训练集和测试机
from sklearn.model_selection import train_test_split
# 引用numpy库,主要用来做科学计算
import numpy as np
# 引用matplotlib库,主要用来画
import matplotlib.pyplot as plt

# 创建数据集,把数据写入到numpy数组
data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
                 [168, 57], [172, 60], [176, 62], [180, 65],
                 [184, 69], [188, 72]])
# 打印出数据大小
print("The size of dataset is (%d,%d)" % data.shape)
# X,y分别存放特征向量和标签. 注:这里使用了reshape(-1,1), 其主要的原因是
# data[:,0]是一维的数组(因为只有一个特征),但后面调用模型的时候对特征向量的要求
# 是矩阵的形式,所以这里做了reshape的操作。
X, y = data[:, 0].reshape(-1, 1), data[:, 1]

# 使用train_test_split函数把数据随机分为训练数据和测试数据。 训练数据的占比由
# 参数train_size来决定。如果这个值等于0.8,就意味着随机提取80%的数据作为训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# TODO 1. 请实例化一个线性回归的模型
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# TODO 2. 在X_train,y_train上训练一个线性回归模型。 如果训练顺利,则regr会存储训练完成之后的结果模型

# 在训练集上做验证,并观察是否训练得当,首先输出训练集上的决定系数R平方值
print("Score for training data %.2f" % regr.score(X_train, y_train))

# 画训练数据
plt.scatter(X_train, y_train, color='red')
# TODO 3. 画在训练数据上已经拟合完毕的直线

# 画测试数据
plt.scatter(X_test, y_test, color='black')
# 画x,y轴的标题
plt.xlabel('height (cm)')
plt.ylabel('weight(kg)')
plt.show()
# 输出在测试集上的决定系数R平方值
print("Score for testing data %.2f" % regr.score(X_test, y_test))
print("Prediction for a person with height 163 is: %.2f" % regr.predict([[163]]))
