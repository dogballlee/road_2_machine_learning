import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


path = r'D:\\py_project\\used_car\\'
train = path + 'used_car_train_20200313.csv'
test = path + 'used_car_testB_20200421.csv'

train_data = pd.read_csv(train, sep=' ')
test_data = pd.read_csv(test, sep=' ')
# print(train_data.shape)
# print(test_data.shape)
n_train_data = train_data.select_dtypes(exclude='object').columns
# print(n_train_data)
feature_col = [col for col in n_train_data if col not in['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]
# print(feature_col)
x_train_data = train_data[feature_col]
y_train_data = train_data['price']

x_test_data = test_data[feature_col]

#定义了一个用均值填充N/A的函数
def fillmean(df):
    for col in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[col].mean()
        df[col].fillna(mean_val,inplace=True)
    return df


x_train_data = fillmean(x_train_data)
x_test_data = fillmean(x_test_data)

xgr = xgb.XGBRegressor(n_estimators=120,learning_rate=.1,gamma=0,subsample=.8,colsample_bytree=.9,max_depth=10)

scores_train = []
scores = []

sk = StratifiedKFold(n_splits=10,shuffle=True,random_state=22)
for train_ind,val_ind in sk.split(x_train_data,y_train_data):
    train_x = x_train_data.iloc[train_ind].values
    train_y = y_train_data.iloc[train_ind]
    val_x = x_train_data.iloc[val_ind].values
    val_y = y_train_data.iloc[val_ind]

    xgr.fit(train_x,train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y,pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y,pred_xgb)
    scores.append(score)

print('train_MAE:',np.mean(scores_train))
print('val_MAE:',np.mean(scores))