# 导入基本的库，每个项目的必备
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib的模式
# matplotlib inline

# 设置matplot的样式
import matplotlib

matplotlib.style.use('ggplot')

# 通过pandas读取.csv文件，并展示头几个样本。
data_df = pd.read_csv('train_subset.csv')
data_df['hour'] = pd.to_datetime(data_df['hour'], format='%y%m%d%H')

data_df = data_df.drop(columns=['id'])
# print(data_df.head())
# print(data_df.iloc[:, :12].head())
# print(data_df.iloc[:, 12:].head())
# print(data_df.info())

# data_df = data_df.set_index('click')
X = data_df.drop(['click'], axis=1)
X = X.values
# print(X)

y = data_df.click.values
# print(y)
# print(data_df.hour.describe())
# print(len(data_df[data_df.click == 0]) / len(data_df))
# print(len(data_df[data_df.click == 1]) / len(data_df))
# print(data_df.hour.describe())

banner_pos = data_df['banner_pos'].unique()

import matplotlib.pyplot as plt

table1 = pd.crosstab(data_df['banner_pos'], data_df['click'])
table1.plot(kind='bar', stacked=True, legend=True, title='Visualization of Banner Position and Click Events')
plt.show()

table1 = table1.div(table1.sum(axis=1), axis=0)

site_features = ['site_id', 'site_domain', 'site_category']
# print(data_df[site_features].describe())

app_features = ['app_id', 'app_domain', 'app_category']
# print(data_df[app_features].describe())

table2 = data_df['app_category'].unique()
table2 = pd.crosstab(data_df['app_category'], data_df['click'])
table2 = table2.div(table2.sum(axis=1), axis=0)
table2.plot(kind='bar', stacked=True, legend=True, title='CTR for app_category feature')
plt.show()

device_features = ['device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']
# print(data_df[device_features].astype('object').describe())

table3 = data_df['device_conn_type'].unique()
table3 = pd.crosstab(data_df['device_conn_type'], data_df['click'])
table3 = table3.div(table3.sum(axis=1), axis=0)
table3.plot(kind='bar', stacked=True, legend=True, title='CTR for device_conn_type feature')
plt.show()

c_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
# print(data_df[c_features].astype('object').describe())

table4 = data_df['C1'].unique()
table4 = pd.crosstab(data_df['C1'], data_df['click'])
table4 = table4.div(table4.sum(axis=1), axis=0)
table4.plot(kind='bar', stacked=True, legend=True, title='CTR for C1 feature')
plt.show()

hour = data_df['hour'].unique()
hour_index = pd.DatetimeIndex(hour)
hour_dict = pd.Series(np.arange(len(hour_index)), index=hour_index)
data_df['hour'].replace(hour_dict, inplace=True)
print(data_df['hour'])

data_df.drop('device_id', axis=1, inplace=True)
data_df.drop('device_ip', axis=1, inplace=True)
data_df.drop('device_model', axis=1, inplace=True)
data_df.drop('site_id', axis=1, inplace=True)
data_df.drop('site_domain', axis=1, inplace=True)
data_df.drop('app_id', axis=1, inplace=True)
print(data_df.astype('object').describe())

data_df = data_df.dropna()
cols = ['hour', 'banner_pos', 'site_category', 'app_domain', 'app_category',
        'device_conn_type', 'C14', 'C18', 'C19', 'C20', 'C21']
for var in cols:
    cat_list = pd.get_dummies(data_df[var], prefix=var)
    data_df = data_df.join(cat_list)

data_df = data_df.drop(cols, axis=1)
print(data_df.columns.values)
feature_names = np.array(data_df.columns[data_df.columns != 'click'].tolist())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data_df[feature_names].values,
    data_df['click'].values,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

params_c = np.logspace(-4, 1, 11)
# TODO: 循环每一个C值，计算交叉验证后的F1-SCORE， 最终选择最好的C值c_best， 然后选出针对这个c_best对应的特征。 务必要使用L1正则。
#       对于实现，有很多方法，自行选择合理的方法就可以了。 关键是包括以下模块：1. 逻辑回归   2. 交叉验证  3. L1正则  4. SelectFromModel
# 先跳过
# from sklearn import linear_model
# logistic = linear_model.LogisticRegression(solver='liblinear', penalty='l1')
# hyperparameters = dict(C=params_c)
# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# best_model = clf.fit(X_train, y_train)

# 求出c_best
# c_best = best_model.best_estimator_.get_params()['C']
c_best = 0.03162277660168379
print('Best C:', c_best)

# 通过c_best值，重新在整个X_train里做训练，并选出特征。
lr_clf = LogisticRegression(penalty='l1', C=c_best, solver='liblinear')
lr_clf.fit(X_train, y_train)  # 在整个训练数据重新训练

select_model = SelectFromModel(lr_clf, prefit=True)
selected_features = select_model.get_support()  # 被选出来的特征

# 重新构造feature_names
feature_names = feature_names[selected_features]

# 重新构造训练数据和测试数据
X_train = X_train[:, selected_features]
X_test = X_test[:, selected_features]

from sklearn.metrics import classification_report  # 这个用来打印最终的结果，包括F1-SCORE

# 暂时也不打开
params_c = np.logspace(-5, 2, 15)  # 也可以自行定义一个范围

# TODO: 实现逻辑回归 + L2正则， 利用GrisSearchCV
from sklearn import linear_model

# logistic = linear_model.LogisticRegression(penalty='l2')
# hyperparameters = dict(C=params_c)
# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# model = clf.fit(X_train, y_train)

# 输出最好的参数
# print(model.best_params_)
#
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

from sklearn.tree import DecisionTreeClassifier

params_min_sampes_split = np.linspace(5, 20, 4).astype(int)
params_min_samples_leaf = np.linspace(2, 10, 5).astype(int)
params_max_depth = np.linspace(4, 10, 4).astype(int)

# dt = DecisionTreeClassifier()
# dt_parameters = dict(
# max_depth=params_max_depth,
# min_samples_split=params_min_sampes_split,
# min_samples_leaf=params_min_samples_leaf
# )
# dt_grid_search_cv = GridSearchCV(dt, dt_parameters, verbose=1)
# model = dt_grid_search_cv.fit(X_train, y_train)
# print(model.best_params_)

# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

params_min_samples_split = np.linspace(5, 20, 4)
# array([ 5., 10., 15., 20.])
params_min_samples_leaf = np.linspace(2, 10, 5)
# array([ 2.,  4.,  6.,  8., 10.])
params_max_depth = np.linspace(4, 10, 4)
# array([ 4.,  6.,  8., 10.])

params_min_sampes_split = np.linspace(5, 20, 4).astype(int)
params_min_samples_leaf = np.linspace(2, 10, 5).astype(int)
params_max_depth = np.linspace(4, 10, 4).astype(int)

pbounds = {
    'min_samples_split': (1, len(params_min_samples_split) + .99),
    'min_samples_leaf': (1, len(params_min_samples_leaf) + .99),
    'max_depth': (1, len(params_max_depth) + .99),
}


def dt_fit(min_samples_split, min_samples_leaf, max_depth):
    score = cross_val_score(DecisionTreeClassifier(
        min_samples_leaf=int(min_samples_leaf) * 5,
        min_samples_split=int(min_samples_split) * 2,
        max_depth=int(max_depth) * 2 + 2
    ), X_train, y_train, scoring='f1', cv=3).mean()
    return score


# Bounded region of parameter space
optimizer = BayesianOptimization(
    f=dt_fit,
    pbounds=pbounds,
    verbose=2,
    random_state=7
)

optimizer.maximize(
    n_iter=10
)

print(optimizer.max)

_dt = DecisionTreeClassifier(
    min_samples_leaf=int(optimizer.max['params']['min_samples_leaf']),
    min_samples_split=int(optimizer.max['params']['min_samples_split']),
    max_depth=int(optimizer.max['params']['max_depth']),
)
model = _dt.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

from xgboost import XGBClassifier

# TODO: 训练XGBoost模型  提示： 使用XGBClassifier。 至于超参数，可以试着去看一下官方文档，然后多尝试尝试。
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# TODO: 在测试数据上预测，并打印在测试集上的结果
clf = XGBClassifier(
    learning_rate=0.1,
    n_estimators=40,
    min_child_weight=6,
    seed=0,
    subsample=0.8,
    reg_alpha=0,
    reg_lambda=1,
    max_depth=5,
    objective='binary:logistic',
)
clf.fit(X_train, y_train, eval_metric='logloss', verbose=True, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))
