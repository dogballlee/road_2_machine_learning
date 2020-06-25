import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import sklearn as sk

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

x_train_data = x_train_data.fillna(-1)
x_test_data = x_test_data.fillna(-1)

xgr = xgb.XGBRegressor(n_estimators=120,learning_rate=.1,gamma=0,subsample=.8,colsample_bytree=.9,max_depth=7)

scores_train = []
scores = []

