#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, KFold
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris =pd.read_csv("~/Downloads/gfarewe/Ytr1.csv")
iris.rename( columns={'Unnamed: 0':'Protein ID'}, inplace=True )
X=iris["Protein ID"]
y=iris["Bound"]
scaler = MinMaxScaler(feature_range=(0, 1))

x_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=3)
kf.get_n_splits(x_scaled)

run = 1
# x_train, x_test, y_train,  y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
for train_index, test_index in kf.split(x_scaled):
    x_train = x_scaled[train_index]
    y_train = y[train_index]

    x_test = x_scaled[test_index]
    y_test = y[test_index]

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    for K in range(1,11):

        model = KNeighborsRegressor(n_neighbors = K)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        error = np.sqrt(mean_squared_error(y_test,prediction)) # Root MSE

        model_scaled = KNeighborsRegressor(n_neighbors = K)
        model_scaled.fit(x_train_scaled, y_train)
        prediction_scaled = model.predict(x_test_scaled)
        error_scaled = np.sqrt(mean_squared_error(y_test,prediction_scaled)) # Root MSE

        print("Fold: %s, K: %s, RMSE: %s, RMSE_Scaled: %s" % (run, K, error, error_scaled))
    run += 1


# In[18]:





# In[ ]:




