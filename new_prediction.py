import numpy as np
import pandas as pd
from sklearn import preprocessing
"""from sklearn import feature_selection"""
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn import linear_model as sk_linear
from sklearn.metrics import mean_squared_error


# 导入数据
path = 'C:/Users/Administrator/Desktop/'
train = pd.read_csv(path + 'zhengqi_train.txt', sep='\t')
test = pd.read_csv(path + 'zhengqi_test.txt', sep='\t')
train_X = train.drop(['target'], axis=1)
train_X.drop(['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)
test.drop(['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)
train_Y = train['target']


# 特征工程
# min_max_scale = preprocessing.minmax_scale
# dataminmax = pd.DataFrame(min_max_scale.fit(all_data), columns=all_data.columns)

pca = decomposition.PCA(n_components=0.97, whiten='True', svd_solver='full')
pca.fit(train_X)
reduced_X = pca.transform(train_X)
reduced_X1 = pca.transform(test)
norm_X = preprocessing.normalize(reduced_X, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(reduced_X, train_Y, test_size=0.20, random_state=42)

# print('PCA:')
# print('降维后的各主成分的方差值占总方差值的比例', pca.explained_variance_ratio_)
# print('降维后的各主成分的方差值', pca.explained_variance_)
# print('降维后的特征数', pca.n_components_)


# 线性回归
model_LR = sk_linear.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
model_LR.fit(X_train, y_train)
y_pred = model_LR.predict(X_test)

print('一般LR误差：', mean_squared_error(y_test, y_pred))

# lasso回归
model_Lasso = sk_linear.LassoLarsCV()
model_Lasso.fit(X_train, y_train)
y_pred_Lasso = model_Lasso.predict(X_test)
print('Lasso误差：', mean_squared_error(y_test, y_pred_Lasso))

# ridge回归
model_ridge = sk_linear.RidgeCV(alphas=[6.9, 7.0, 7.1])
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
print('ridge误差：', mean_squared_error(y_test, y_pred_ridge))
print('alpha值：', model_ridge.alpha_)

print('MSE:', mean_squared_error(y_test, y_pred))


# 由于使用ridge时MSE最小，故选取该方法（去求吧，好像过拟合了............还是一般LR好使）

# print('线性回归:')
# print('截距:',model.intercept_) #输出截距
# print('系数:',model.coef_) #输出系数


test_set = np.array(reduced_X1)

par = np.array(model_LR.coef_)


# print('test_set', test_set)
# print('par', par)
L = np.matmul(test_set, par)
df = pd.DataFrame(L)
df.to_csv(path + 'zhengqi_predict.txt', index=0, header=0, sep='\t')
