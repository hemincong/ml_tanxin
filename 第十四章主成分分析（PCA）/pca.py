import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 直接从sklearn导入iris数据
data = load_iris()
# 读取特征、标签
X = data.data
y = data.target
# TODO 做数据的归一化(必要的步骤!!!)
X_scaled = StandardScaler().fit_transform(X, y)
# TODO: 计算数据的covariance矩阵  cov = (1/n)* X'*X  (X' 是 X的转置), n为样本总数
cov_matrix = np.matmul(X_scaled.T, X_scaled) / len(X_scaled)
# 根据cov_matrix计算出特征值与特征向量,调用linalg.eig函数
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
# eigenvalues存放所有计算出的特征值,从大到小的顺序
# eigenvectors存放所有对应的特征向量,这里 eigenvectors[:,0]表示对应的第一个特征向量
#    注:是列向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# 根据计算好的特征向量,计算每个向量的重要性
import matplotlib.pyplot as plt

explained_variances = []
for i in range(len(eigenvalues)):
    explained_variances.append(eigenvalues[i] / np.sum(eigenvalues))
plt.plot(range(1, 5), np.array(explained_variances).cumsum())
plt.title('Explained Variance', fontsize=15)
plt.xlabel('Number of Principle Components', fontsize=10)
plt.show()
# 把数据映射到二维的空间(使用前两个特征向量)
pca_project_1 = X_scaled.dot(eigenvectors.T[0])  # 基于第一个特征向量的维度值
pca_project_2 = X_scaled.dot(eigenvectors.T[1])  # 基于第二个特征向量的维度值
# 构造新的二维数据
res = pd.DataFrame(pca_project_1, columns=['PCA_dim1'])
res['PCA_dim2'] = pca_project_2
res['Y'] = y
print(res.head())
import seaborn as sns

# 仅用第一个维度做可视化,从结果中会发现其实已经分开的比较好了
plt.figure(figsize=(20, 10))
sns.scatterplot(res['PCA_dim1'], [0] * len(res), hue=res['Y'], s=200)
# 用前两个维度做可视化,从结果中发现分开的比较好
plt.figure(figsize=(20, 10))
sns.scatterplot(res['PCA_dim1'], res['PCA_dim2'], hue=res['Y'], s=200)
plt.show()
