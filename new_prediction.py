import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition

#导入数据
path = 'C:/Users/Administrator/Desktop/'
train = pd.read_csv(path + 'zhengqi_train.txt', sep='\t')
test = pd.read_csv(path + 'zhengqi_test.txt', sep='\t')
train_X = train.drop(['target'],axis=1)
all_data = pd.concat([train_X, test])

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