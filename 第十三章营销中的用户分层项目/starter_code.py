import datetime as dt

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_excel('OnlineClean.xlsx', engine='openpyxl', )
print(df.head())

df['InvoiceDate'] = df['InvoiceDate'].dt.date
print('Min date = {}, Max data = {}'.format(min(df.InvoiceDate), max(df.InvoiceDate)))

df = df[df['InvoiceDate'] > dt.date(2010, 12, 9)]
print(df.head())

snapshot_date = max(df.InvoiceDate) + dt.timedelta(days=1)
print(snapshot_date)

df['TotalSum'] = df['Quantity'] * df['UnitPrice']
print(df.head())

data_rfm = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum',
}).sort_index(ascending=True)

data_rfm.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue',
                         }, inplace=True)
print(data_rfm.head())

Rquartiles = pd.qcut(data_rfm['Recency'], 4, labels=range(4, 0, -1))
data_rfm = data_rfm.assign(Recency_Q=Rquartiles.values)

Fquartiles = pd.qcut(data_rfm['Frequency'], 4, labels=range(1, 5))
data_rfm = data_rfm.assign(Frequency_Q=Fquartiles.values)

Mquartiles = pd.qcut(data_rfm['MonetaryValue'], 4, labels=range(1, 5))
data_rfm = data_rfm.assign(Moneytary_Q=Mquartiles.values)

print(data_rfm.head())

data_rfm['Segment'] = data_rfm[['Recency_Q', 'Frequency_Q', 'Moneytary_Q']].apply(lambda x: ''.join(x.map(str)), axis=1)
print(data_rfm.head())

data_rfm['RFM_Score'] = data_rfm[['Recency_Q', 'Frequency_Q', 'Moneytary_Q']].sum(axis=1)
print(data_rfm.head())

RFM_Cluster = pd.qcut(data_rfm['RFM_Score'], 3, labels=['Bronze', 'Silver', 'Gold'])
data_rfm = data_rfm.assign(RFM_Lavel=RFM_Cluster.values)
print(data_rfm.head())

df_rfm_custom_segment = data_rfm.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
}).round(1)

print(df_rfm_custom_segment)

# 只选取R,F,M的值
data_rfm = data_rfm[['Recency', 'Frequency', 'MonetaryValue']]

# 1. 先做log transform, +1是为了避免log(0)
data_rfm = np.log(data_rfm + 1)

# 2. 归一化操作
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_rfm)
data_rfm = scaler.transform(data_rfm)

print(data_rfm[0:5])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

sse = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=99).fit(data_rfm)
    cluster_labes = kmeans.labels_
    sse[k] = kmeans.inertia_

plt.title('Distortion Score Elbow for KMeans')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

kmeans_3 = KMeans(n_clusters=3, random_state=100).fit(data_rfm)
cluster_labels_3 = kmeans_3.labels_
average_rfm_3 = kmeans_3.cluster_centers_
print(average_rfm_3)

kmeans_4 = KMeans(n_clusters=4, random_state=100).fit(data_rfm)
cluster_labels_4 = kmeans_4.labels_
average_rfm_4 = kmeans_4.cluster_centers_
print(average_rfm_4)

kmeans_5 = KMeans(n_clusters=5, random_state=100).fit(data_rfm)
cluster_labels_5 = kmeans_5.labels_
average_rfm_5 = kmeans_5.cluster_centers_
print(average_rfm_5)
