# 请在下方作答
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data_df = pd.read_csv('insurance.csv')
feature_names = np.array(['age', 'bmi', 'children'])
X_train, X_test, y_train, y_test = train_test_split(
    data_df[feature_names].values,
    data_df['charges'].values,
    train_size=0.75,
    test_size=0.25,
    random_state=1
)
lr = LinearRegression().fit(X_train, y_train)

print(lr.score(X_test, y_test))

forcast = lr.predict(X_test)  # 预测

liner_int = lr.intercept_
print(liner_int)
liner_cof = lr.coef_
print(liner_cof)
R2 = r2_score(y_test, forcast)
print(R2)
#评价选项：_____