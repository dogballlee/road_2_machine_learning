import pandas as pd
import numpy as np
import math
import sklearn as sk
import seaborn as sb
import matplotlib.pyplot as plt

path = 'C:/Users/Administrator/Desktop/'

train = pd.read_csv(path + 'zhengqi_train.txt', sep='\t')
test = pd.read_csv(path + 'zhengqi_test.txt', sep='\t')
train_x = train.drop(['target'],axis=1)
all_data = pd.concat([train_x, test])
all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

min_max_scaler = sk.preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data), columns=all_data.columns)








