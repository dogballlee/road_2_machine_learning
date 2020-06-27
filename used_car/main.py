import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
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

# xgr = xgb.XGBRegressor(n_estimators=120,learning_rate=.1,gamma=0,subsample=.8,colsample_bytree=.9,max_depth=10)
#
# scores_train = []
# scores = []
#
# sk = StratifiedKFold(n_splits=10,shuffle=True,random_state=22)
# for train_ind,val_ind in sk.split(x_train_data,y_train_data):
#     train_x = x_train_data.iloc[train_ind].values
#     train_y = y_train_data.iloc[train_ind]
#     val_x = x_train_data.iloc[val_ind].values
#     val_y = y_train_data.iloc[val_ind]
#
#     xgr.fit(train_x,train_y)
#     pred_train_xgb = xgr.predict(train_x)
#     pred_xgb = xgr.predict(val_x)
#
#     score_train = mean_absolute_error(train_y,pred_train_xgb)
#     scores_train.append(score_train)
#     score = mean_absolute_error(val_y,pred_xgb)
#     scores.append(score)
#
# print('train_MAE:',np.mean(scores_train))
# print('val_MAE:',np.mean(scores))



def build_model_xgb(x_train,y_train):
    model_xgb = xgb.XGBRegressor(n_estimators=120,learning_rate=.1,gamma=0,subsample=.8,colsample_bytree=.9,max_depth=10)
    model_xgb.fit(x_train,y_train)
    return model_xgb

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=150,n_estimators=150)
    param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
    gbm = GridSearchCV(estimator,param_grid)
    gbm.fit(x_train,y_train)
    return gbm


x_train,x_val,y_train,y_val = train_test_split(x_train_data,y_train_data,test_size=.3)


#模型训练
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val,val_lgb)
print('MAE of val with lgb:',MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(x_train_data,y_train_data)
subA_lgb = model_lgb_pre.predict(x_test_data)
print('Sta of Predict lgb:')

print('Train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val,val_xgb)
print('MAE of val with xgb:',MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(x_train_data,y_train_data)
subA_xgb = model_xgb_pre.predict(x_test_data)
print('Sta of Predict xgb:')


#xgb与lgb加权融合
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0]=10 # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,val_Weighted))

sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb


sub = pd.DataFrame()
sub['SaleID'] = test_data.SaleID
sub['price'] = sub_Weighted
sub.to_csv('./sub_Weighted.csv',index=False)