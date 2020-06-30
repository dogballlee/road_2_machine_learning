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
feature_col = [col for col in n_train_data if col not in['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]

x_train_data = train_data[feature_col]
x_test_data = test_data[feature_col]
x_train_data = x_train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
x_test_data = x_test_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
y_train_data = train_data['price']

# print(x_train_data.shape)   (150000, 21)
# print(y_train_data.shape)   (150000,)
# print(x_test_data.shape)    (50000, 21)

for col in feature_col:
    g = sns.kdeplot(x_train_data[col],color='red',shade=True)
    g = sns.kdeplot(x_test_data[col], color='blue',ax=g, shade=True)
    plt.show()

