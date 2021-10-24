import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def  cls(): print("\n"*20)
# %matplotlib inline: Use plt.show for Python

def bundleClassifier():
    pass

# ***************************************

def kpredictor():
    model = sklearn.linear_model.HuberRegressor()
    model.max_iter = 20000
    splitRatio = 0.1

def defaultCode():
    models = dir(sklearn.linear_model)
    rawinputdata = pd.read_csv('data21.csv')
    x = rawinputdata[[i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Pref (primary)'or i=='Prefer optimization' or i=='Pref (second)')]]
    y = rawinputdata['k']
    models.remove('ElasticNet')
    models.remove('ElasticNetCV')
    models.remove('Hinge')
    models.remove('Huber')
    models.remove('Lars')
    models.remove('LarsCV')
    models.remove('LassoCV')
    models.remove('Lasso')
    models.remove('LassoLarsCV')
    models.remove('LassoLars')
    models.remove('LassoLarsIC')
    models.remove('Log')
    
    for _model in models:
        all_mses = []
        all_r2s = []
        print("**********\n")
        splitRatio = 0.1
        print("Current Model:   ",_model)
        for i in range(9):
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitRatio,random_state = 0)
            # model=LinearRegression()
            model = getattr(sklearn.linear_model,_model)()
            model.max_iter = 20000
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            # x = inputdata[]
            all_mses.append(mean_squared_error(y_test,y_pred))
            all_r2s.append(r2_score(y_test,y_pred))
            splitRatio+=0.1
            #print("Current splitRatio ",splitRatio)
        print("Best R2, max: ", max(all_r2s), "occuring at splitRatio = ",0.1+0.1*all_r2s.index(max(all_r2s)))
        print("Best MSE, min", min(all_mses), "occuring at splitRatio = ",0.1+0.1*all_mses.index(min(all_mses)))

        # from sklearn.linear_model import LinearRegression
        # from sklearn.metrics import r2_score
        # model=LinearRegression()
        # model.fit(x_train,y_train)
        # y_pred=model.predict(x_test)
defaultCode()

def runClassifier():
    pass
def newBundleRun():
    pass


if __name__ == "__main__":
    shouldClassify = ''
    while not (shouldClassify in ['1','2']):
        shouldClassify = input("Enter '1' for existing bundle recommendation and '2' for new bundle generator: \n")
    if (shouldClassify == '1'):
        runClassifier()
    else: #shouldClassify is 2
        newBundleRun()
