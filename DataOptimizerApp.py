import pandas as pd
#import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def  cls(): print("\n"*20)
# %matplotlib inline: Use plt.show for Python

def bundleClassifier(_inputParams):
    import sklearn.preprocessing
    model = sklearn.svm.SVC()
    model.max_iter = 20000
    splitRatio = 0.1
    rawinputdata = pd.read_csv('data21.csv')
    x = rawinputdata[[i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Pref (primary)'or i=='Prefer optimization' or i=='Pref (second)' or i=='Unnamed')]]
    sklearn.preprocessing.StandardScaler().fit(x)
    y = rawinputdata['Pref (primary)']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitRatio,random_state = 0)
    model.fit(x_train, y_train)
    inputParams = pd.DataFrame(data=_inputParams)
    result = model.predict(inputParams).__getitem__(0)
    print("bundleClassifier result is {}".format(result))
    existingBundles = []
    return(existingBundles[result])

# ***************************************

def kpredictor(_inputParams):
    model = sklearn.linear_model.HuberRegressor()
    model.max_iter = 20000
    splitRatio = 0.1
    rawinputdata = pd.read_csv('data21.csv')
    x = rawinputdata[[i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Pref (primary)'or i=='Prefer optimization' or i=='Pref (second)' or i=='Unnamed')]]
    y = rawinputdata['k']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitRatio,random_state = 0)
    model=sklearn.linear_model.LinearRegression()
    model.fit(x_train, y_train)
    inputParams = pd.DataFrame(data=_inputParams)
    result = model.predict(inputParams)
    print("kpredictor result is {}".format(result))
    return(result)
    
# def defaultCode():
if __name__ == "__maxin__":
    models = dir(sklearn.linear_model)
    rawinputdata = pd.read_csv('data21.csv')
    x = rawinputdata[[i for i in list(rawinputdata.columns) if not (i=='k' or i=='Purchases /month' or i=='Pref (primary)'or i=='Prefer optimization' or i=='Pref (second)' or i=='Unnamed')]]
    y = rawinputdata['k']
    # print("y the input is \n",y)
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
        print("\n\n\n**********************************************")
        splitRatio = 0.1
        print("Current Model:   ",_model)
        for i in range(1):
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = splitRatio,random_state = 0)
            model=sklearn.linear_model.LinearRegression()
            # model = getattr(sklearn.linear_model,_model)()
            print(model)
            model.max_iter = 20000
            model.fit(x_train, y_train)
            x_test2 = pd.DataFrame(data=['4', '1', '0', '10', '3', '2', '5', '3', '8', '2', '0.006359975'])
            print("test data format:\n",x_test2,"\n of type: \n",type(x_test))
            y_pred = model.predict(x_test2)
            # y_pred=model.predict(x_test,None,x.columns)
            print("prediction: ",y_pred)
            # x = inputdata[]
            all_mses.append(mean_squared_error(y_test,y_pred))
            all_r2s.append(r2_score(y_test,y_pred))
            splitRatio+=0.1
            #print("Current splitRatio ",splitRatio)
##        print("Best R2, max: ", max(all_r2s), "occuring at splitRatio = ",0.1+0.1*all_r2s.index(max(all_r2s)))
##        print("Best MSE, min", min(all_mses), "occuring at splitRatio = ",0.1+0.1*all_mses.index(min(all_mses)))

        # from sklearn.linear_model import LinearRegression
        # from sklearn.metrics import r2_score
        # model=LinearRegression()
        # model.fit(x_train,y_train)
        # y_pred=model.predict(x_test)
# defaultCode()

def runClassifier(inputParams, ):
    return bundleClassifier(inputParams)

def newBundleRun(inputParams, userMaxMB):
    k = kpredictor(inputParams)
    bundleMB = max(20, int(userMaxMB))
    price = k*bundleMB
    return '{} MB at GHC {}'.format(bundleMB, price)


if __name__ == "__main__":
    shouldClassify = ''
    while not (shouldClassify in ['1','2']):
        shouldClassify = input("Enter '1' for existing bundle recommendation and '2' for new bundle generator: \n")
    #Data Collection:
    inputParams = {}
    input_columns = ['WhatsApp', 'YouTube', 'FB', 'Twitter', 'News', 'Research','Coursera/edX', 'Webinar', 'Email', 'High social media', 'Games']
    print("User data collection on usage parameters:\n************")
    for i,v in enumerate(input_columns):
        inputParams[v] = [float(input("{column:<}:  ".format(column=v)))]
    result = ""
    if (shouldClassify == '1'):
        result = runClassifier(inputParams)
    else: #shouldClassify is 2
        maxMBtoColumn = {'WhatsApp': 300,'YouTube': 2048, 'FB': 1024, 'Twitter': 1024, 'News': 300, 'Research': 300, 'Coursera/edX': 1024, 'Webinar': 512, 'Email': 300,'High social media': 2048, 'Games': 512}
        userMaxMB = maxMBtoColumn[max(inputParams, key=inputParams.get)]
        result = newBundleRun(inputParams, userMaxMB)
        
    print("Resulting bundle is: ")
    print(result)