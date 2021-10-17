import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %matplotlib inline: Use plt.show for Python
rawinputdata = pd.read_csv('data.csv')
splitRatio = 0.1

x = rawinputdata[[i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Pref (primary)'or i=='Prefer optimization' or i=='Pref (second)')]]
y = rawinputdata['k']
# print(dir(x))
# print([i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Prefer optimization' or i=='Pref (second)')])
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitRatio,random_state = 0)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# x = inputdata[]


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# model=LinearRegression()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)