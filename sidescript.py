import pandas as pd
import math

# definitions = [None,10/971.82,12/(2.5*1024),None,5/1024,3/471.7,10/616.32,5/292.4]

# csv = pd.read_csv('data.csv')

##for i,v in enumerate(csv['k']):
##	print("***",i)
##	print(csv['k'][i])
##	csv['k'][i] = definitions[int(csv['Pref (primary)'][i])]
##
##print(csv)
##csv.to_csv('data21.csv')
### print(csv['k'])

def returner():
    return (1,2)

result = returner()
print(result)

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