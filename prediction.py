import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import seaborn
import matplotlib.pyplot as plt

path = 'C:/Users/Administrator/Desktop/'

#导入数据，剔除异常样本
train = pd.read_csv(path + 'zhengqi_train.txt', sep='\t')
test = pd.read_csv(path + 'zhengqi_test.txt', sep='\t')
train_x = train.drop(['target'],axis=1)
all_data = pd.concat([train_x, test])
all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

#归一化数据
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data), columns=all_data.columns)

#观察当前数据分布情况
# for col in data_minmax.columns:
#     seaborn.distplot(data_minmax[col])
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     plt.show()

#将非正态分布数据转换为正态
data_minmax['V0'] = data_minmax['V0'].apply(lambda x:math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x:math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x:math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])
#train['exp'] = train['target'].apply(lambda x:math.pow(1.5,x)+10)

X_scaled = pd.DataFrame(preprocessing.scale(data_minmax),columns = data_minmax.columns)
train_x = X_scaled.ix[0:len(train)-1]
test = X_scaled.ix[len(train):]
Y=train['target']

#特征选择
threshold = 0.85
vt = VarianceThreshold().fit(train_x)
# Find feature names
feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1-threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]

#单变量
X_scored = SelectKBest(score_func=f_regression, k='all').fit(train_x, Y)
feature_scoring = pd.DataFrame({'feature': train_x.columns,'score': X_scored.scores_})
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)