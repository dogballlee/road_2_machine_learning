import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition
from sklearn import linear_model as sk_linear
from sklearn.model_selection import train_test_split

#导入数据
path = 'C:/Users/Administrator/Desktop/'
train = pd.read_csv(path + 'zhengqi_train.txt', sep='\t')
test = pd.read_csv(path + 'zhengqi_test.txt', sep='\t')
train_X = train.drop(['target'],axis=1)
train_Y = train['target']
all_data = pd.concat([train_X, test])
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)

#特征工程
#min_max_scale = preprocessing.minmax_scale
#dataminmax = pd.DataFrame(min_max_scale.fit(all_data), columns=all_data.columns)
pca = decomposition.PCA(n_components='mle', whiten='True', svd_solver='full')
pca.fit(train_X)
reduced_X = pca.transform(train_X) #为降维后的数据
"""
print('PCA:')
print('降维后的各主成分的方差值占总方差值的比例', pca.explained_variance_ratio_)
print('降维后的各主成分的方差值', pca.explained_variance_)
print('降维后的特征数', pca.n_components_)
"""

model = sk_linear.LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
model.fit(X_train, y_train)
model.predict(X_test)
acc=model.score(X_test, y_test) #返回预测的确定系数R2
"""
print('线性回归:')
print('截距:',model.intercept_) #输出截距
print('系数:',model.coef_) #输出系数
print('线性回归模型评价:',acc)
"""

test_set = np.array(test)
par = np.array(model.coef_)
#print('test_set', test_set)
#print('par', par)
L = np.matmul(test_set, par)
df = pd.DataFrame(L)
df.to_csv(path + 'zhengqi_predict.txt',index=0 ,header=0 , sep='\t')
#print(L)
