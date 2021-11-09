import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.linear_model

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
    existingBundles = ['971.82 MB at GHC 10', '2.5 GB at GHC 12', 'Midnight Bundle', '1GB at GHC 5 (Just for Me)', 'Special bundle, 471.70 MB at GHC 3', 'MTN Pulse at GHC 5', 'MTN Pulse at GHC 10']
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
    print("kpredictor result is {} of type{}".format(result, type(result)) )
    return(result)
    


def runClassifier(inputParams, ):
    return bundleClassifier(inputParams)

def newBundleRun(inputParams, userMaxMB):
    k = kpredictor(inputParams)
    bundleMB = max(20, int(userMaxMB))
    price = (k*bundleMB)
    roundPrice = (k*bundleMB)[0].__round__()
    bundleMB = bundleMB*(price/roundPrice)
    
    return 'Newly-generated bundle:\n{} MB at GHC {}'.format(bundleMB[0], roundPrice)


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