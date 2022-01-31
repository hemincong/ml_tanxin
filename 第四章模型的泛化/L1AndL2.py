# 随机生成样本数据。 二分类问题,每一个类别生成5000个样本数据
import numpy as np
np.random.seed(12)
num_observations = 100 # 生成正负样本各100个
# 利用高斯分布来生成样本,首先需要生成covariance matrix
# 由于假设我们生成20维的特征向量,所以矩阵大小为20*20
rand_m = np.random.rand(20,20)
# 保证矩阵为PSD矩阵(半正定)
cov = np.matmul(rand_m.T, rand_m)
# 通过高斯分布生成样本
x1 = np.random.multivariate_normal(np.random.rand(20), cov, num_observations)
x2 = np.random.multivariate_normal(np.random.rand(20)+5, cov, num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
from sklearn.linear_model import LogisticRegression
# 使用L1的正则,C为控制正则的参数。C值越大,正则项的强度会越弱。
clf = LogisticRegression(fit_intercept=True, C=0.1, penalty='l1')
clf.fit(X, y)
print ("(L1)逻辑回归的参数w为: ", clf.coef_)
# 使用L2的正则,C为控制正则的参数。C值越大,正则项的强度就会越弱
clf = LogisticRegression(fit_intercept=True, C=0.1, penalty='l2')
clf.fit(X, y)
print ("(L2)逻辑回归的参数w为: ", clf.coef_)
