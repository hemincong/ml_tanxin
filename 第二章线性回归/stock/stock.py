# 导入必要的库
import numpy as np  # 数学计算
import pandas as pd  # 数据处理, 读取 CSV 文件 (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # 可视化工具
from datetime import datetime as dt  # 时间的工具
from sklearn import preprocessing  # 归一化时用到

df = pd.read_csv('./000001.csv')

print(np.shape(df))
print(df.head())

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# TODO: 按照时间升序排列数据, 使用df中的sort_values函数
#
#
df.sort_values(by=['date'], ascending=False)
print(df.head(10))

df.dropna(axis=0, inplace=True)
df.isna().sum()

min_date = df.index.min()
max_date = df.index.max()
print("First date is", min_date)
print("Last date is", max_date)

num = 5  # 预测5天后的情况
df['label'] = df['close'].shift(-num)
print(df.shape)
print(df.tail(10))

df.dropna(inplace=True)
print(df.tail(10))

X = df.drop(['price_change', 'label', 'p_change'], axis=1)
X = X.values

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

y = df.label.values
print(np.shape(X), np.shape(y))

X_train, y_train = X[0:550, :], y[0:550]
X_test, y_test = X[550:606, :], y[550:606]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


lr = LinearRegression().fit(X_train, y_train)

print(lr.score(X_test, y_test))  # 使用绝对系数 R^2 评估模型

x_predict = X[-100:]  # 选取最新的100个样本
forcast = lr.predict(x_predict)  # 预测

plt.plot(y[-100:], color='r', label="actual value")
plt.plot(forcast, color='b', label="predicted value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
