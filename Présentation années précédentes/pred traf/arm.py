import os
path = "/home/n7student/Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/"
os.path.dirname(path)
print(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pmdarima
import math

df1 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario1/tx_throughput.csv',sep =';',header=0, index_col=0)
df2 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario2/tx_throughput.csv',sep =';',header=0, index_col=0)
df3 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario3/tx_throughput.csv',sep =';',header=0, index_col=0)
df4 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario4/tx_throughput.csv',sep =';',header=0, index_col=0)
df5 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario5/tx_throughput.csv',sep =';',header=0, index_col=0)
df6 =pd.read_csv('Bureau/PRED_TRAFFIC/nb_variable_utilisateurs/scenario6/tx_throughput.csv',sep =';',header=0, index_col=0)




size = int(len(X)*0.6)
train,test = X[0:size], X[size:len(X)]
model = ARIMA(train,order=(4,0,2))
model_fit=model.fit()
print(len(train))
print(len(test))
pred = model_fit.predict(start=len(train),end=len(test)+len(train)-1)
plt.plot(pred,'r')
plt.plot(X)
plt.show()

#plt.plot(df2)
#plt.show()
#plt.plot(df3)
#plt.show()
#plt.plot(df4)
#plt.show()
#plt.plot(df5)
#plt.show()
#plt.plot(df6)
#plt.show()