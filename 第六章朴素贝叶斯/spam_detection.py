import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 读取spam.csv文件
df = pd.read_csv("spam.csv", encoding='latin')
df.head()
# 重命名数据中的v1和v2列,使得拥有更好的可读性
df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
df.head()
# 把'ham'和'spam'标签重新命名为数字0和1
df['numLabel'] = df['Label'].map({'ham': 0, 'spam': 1})
df.head()
# 统计有多少个ham,有多少个spam
print("# of ham : ", len(df[df.numLabel == 0]), " # of spam: ", len(df[df.numLabel == 1]))
print("# of total samples: ", len(df))
# 统计文本的长度信息,并画出一个histogram
text_lengths = [len(df.loc[i, 'Text']) for i in range(len(df))]
plt.hist(text_lengths, 100, facecolor='blue', alpha=0.5)
plt.xlim([0, 200])
plt.show()
# 导入英文的停用词库
from sklearn.feature_extraction.text import CountVectorizer

# 构建文本的向量 (基于词频的表示)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
y = df.numLabel
# 把数据分成训练数据和测试数据
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
print("训练数据中的样本个数: ", X_train.shape[0], "测试数据中的样本个数: ", X_test.shape[0])
# 利用朴素贝叶斯做训练
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data: ", accuracy_score(y_test, y_pred))
# 打印混淆矩阵
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred, labels=[0, 1])
