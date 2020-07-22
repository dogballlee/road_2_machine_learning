import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



path = r'D:\\py_project\\used_car\\'
train = path + 'used_car_train_20200313.csv'
test = path + 'used_car_testB_20200421.csv'

train_data = pd.read_csv(train, sep= ' ')
test_data = pd.read_csv(test, sep=' ')
n_train_data = train_data.select_dtypes(exclude='object').columns
feature_col = [col for col in n_train_data if col not in['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller','offerType','v_13','v_14']]
# print(len(feature_col))

x_train_data = train_data[feature_col]
x_test_data = test_data[feature_col]
# X_train_scaled = x_train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# X_test_scaled = x_test_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        #min_max标准化
X_train_scaled = x_train_data.apply(lambda x: (x - np.mean(x)) / np.std(x))
X_test_scaled = x_test_data.apply(lambda x: (x - np.mean(x)) / np.std(x))      #z-score标准化
y_train_data = train_data['price']

# print(x_train_data.shape)   (150000, 21)
# print(y_train_data.shape)   (150000,)
# print(x_test_data.shape)    (50000, 21)

for col in feature_col:
    plt.subplot(3,6,feature_col.index(col)+1)
    g = sns.kdeplot(X_train_scaled[col],color='red',shade=True)
    g = sns.kdeplot(X_test_scaled[col], color='blue',ax=g, shade=True)
plt.show()