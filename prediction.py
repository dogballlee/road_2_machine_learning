import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
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

n_folds = 10


def rmsle_cv(model, train_x_head=train_x_head):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    rmse = -cross_val_score(model, train_x_head, Y, scoring="neg_mean_squared_error", cv=kf)
    return (rmse)


svr = make_pipeline(SVR(kernel='linear'))

line = make_pipeline(LinearRegression())
lasso = make_pipeline(Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
# KRR3 = KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
# =============================================================================
# GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.02,
#                                    max_depth=5, max_features=7,
#                                    min_samples_leaf=15, min_samples_split=10,
#                                    loss='huber', random_state =5)
# =============================================================================

model_xgb = xgb.XGBRegressor(booster='gbtree', colsample_bytree=0.8, gamma=0.1,
                             learning_rate=0.02, max_depth=5,
                             n_estimators=500, min_child_weight=0.8,
                             reg_alpha=0, reg_lambda=1,
                             subsample=0.8, silent=1,
                             random_state=42, nthread=2)

# =============================================================================
# cv_params = {'min_child_weight': [0.05,0.1,0.15,0.2,0.25],
#              'learning_rate': [0.01, 0.02, 0.05, 0.1],
#              'max_depth': [3,5,7,9]}
#
# other_params = {'learning_rate': 0.02, 'n_estimators': 400, 'max_depth': 5, 'min_child_weight': 0.8, 'seed': 0,
#                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1}
#
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(train_x, Y)
# evalute_result = optimized_GBM.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
# model_xgb = xgb.XGBRegressor(optimized_GBM.best_params_)
# =============================================================================
# =============================================================================
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# =============================================================================


# =============================================================================
# parameters = {
#             'n_estimators':[300,600,900,1500,2500],
#             #'boosting':'dart',
#             'max_bin':[55,75,95],
#             'num_iterations':[50,100,250,400],
#              # 'max_features':[7,9,11,13],
#               'min_samples_leaf': [15, 25, 35, 45],
#               'learning_rate': [0.01, 0.03, 0.05, 0.1],
#               'num_leaves':[15,31,63],
#
#               'lambda_l2':[0,1]}  # 定义要优化的参数信息
# clf = GridSearchCV( model_lgb, parameters, n_jobs=3,scoring = 'neg_mean_squared_error' )
# clf.fit(train_x,Y)
# =============================================================================


# print('best n_estimators:', clf.best_params_)
# print('best cv score:', clf.score_)


score = rmsle_cv(svr)
print("\nSVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
svr.fit(train_x_head, Y)
score = rmsle_cv(line)
print("\nLine 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lasso)
print("\nLasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR2)
print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR2.fit(train_x_head, Y)
# score = rmsle_cv(KRR3)
# print("Kernel Ridge3 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# =============================================================================
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head2 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)
score = rmsle_cv(KRR1, train_x_head2)
print("Kernel Ridge1 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("Gradient Boosting 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# =============================================================================
head_feature_num = 22
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head3 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x), columns=train_x.columns)
score = rmsle_cv(model_xgb, train_x_head3)
print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(train_x_head, Y)
# =============================================================================
# score = rmsle_cv(model_lgb)
# print("LGBM 得分: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# =============================================================================

# 简单模型融合
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 遍历所有模型，拟合数据
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # 预估，并对预估结果值做average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # return 0.85*predictions[:,0]+0.15*predictions[:,1]
        # return 0.7*predictions[:,0]+0.15*predictions[:,1]+0.15*predictions[:,2]
        return np.mean(predictions, axis=1)
        # averaged_models = AveragingModels(models = (lasso,KRR))


averaged_models = AveragingModels(models=(svr, KRR2, model_xgb))

score = rmsle_cv(averaged_models)
print(" 对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))