import pandas as pd
import math
from functools import reduce
# from pandas import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 
from statsmodels.tsa.arima_model import ARIMA

#Note - need to run python -m pip install statsmodels  and pip install yfinance

def downloadData(ticker, start, end):
    # df = yf.download('TSLA', start='2019-01-01', end='2019-12-31', progress=False)
    df = yf.download(ticker, start, end)
    df["Date"] = df.index
    return df


def processInput(input):
   
    numRows = input.shape[0]     #not counting labels
    numCols = input.shape[1] - 2 #not counting Number and Date

    TEST_OFFSET = 10
    x = input["Date"]
    y = input["Close"]
    x_linreg = x.map(pd.datetime.toordinal)

    x_train = np.asarray(x[0:numRows - TEST_OFFSET ]).reshape(-1,1)
    y_train = np.asarray(y[0:numRows - TEST_OFFSET]).reshape(-1,1)
    x_test = np.asarray(x[numRows - TEST_OFFSET:numRows+1]).reshape(-1,1)
    y_test = np.asarray(y[numRows - TEST_OFFSET:numRows+1]).reshape(-1,1)

    x_train_linreg = np.asarray(x_linreg[0:numRows - TEST_OFFSET ]).reshape(-1,1)
    x_test_linreg = np.asarray(x_linreg[numRows - TEST_OFFSET:numRows+1]).reshape(-1,1)

    return x_train, y_train, x_test, y_test, x_train_linreg,  x_test_linreg
    

def linreg(x_train, y_train, x_test, y_test):
    model = LinearRegression().fit(x_train, y_train)
    # print(model.coef_, model.intercept_)

    y_pred = model.predict(x_test)
    error = math.sqrt(mean_squared_error(y_test, y_pred))
    # print('Mean squared error: %.2f' % error)
    return y_pred, error

def arima(x_train, y_train, x_test, y_test, arimaOrder = (1,1,1)):
    historical = [i for i in y_train]
    errors = list()
    #predict future len(y_test) observations
    predictions = list()
    for t in range(len(y_test)):
        model = ARIMA(historical, order=arimaOrder)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        y_pred = output[0]
        predictions.append(y_pred)
        obs = y_test[t]
        historical.append(obs)
        # print('predicted=%f, expected=%f' % (y_pred, obs))
        error = math.sqrt(mean_squared_error(y_test[t], y_pred))
        errors.append(error)
    #get final error
    finalError = math.sqrt(mean_squared_error(y_test, predictions))
    return predictions, finalError

def arimaGridSearch(x_train, y_train, x_test, y_test, ps, ds, qs):
    minError, bestOrder = float("inf"), None
    bestPreds = None
    for p in ps:
        for d in ds:
            for q in qs:
                arimaOrder = (p,d,q)
                try:    #prevent ValueError if coefficients not appropriate
                    preds, error = arima(x_train, y_train, x_test, y_test, arimaOrder)
                    if minError > error:
                        minError = error
                        bestOrder = arimaOrder
                        bestPreds = [pred for pred in preds]
                except:
                    continue    #skip if coeffs give error
    return bestPreds, minError, bestOrder

def knn(x_train, y_train, x_test, y_test, k):
    #re-scale features to 0->1 range
    mmscaler = MinMaxScaler(feature_range=(0,1))
    x_train = mmscaler.fit_transform(x_train)
    x_test = mmscaler.fit_transform(x_test)

    #build model
    model = neighbors.KNeighborsRegressor(n_neighbors = k)
    model_fit = model.fit(x_train, y_train)  #fit the model
    y_pred = model.predict(x_test)
    #rmse error
    error = math.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, error

def knnOptim(x_train, y_train, x_test, y_test, maxK):
    minError = float("inf")
    bestPreds = None
    bestK = 1
    for k in range(1, maxK+1):
        preds, error = knn(x_train, y_train, x_test, y_test, k)
        # print(preds, error)
        if minError > error:
            minError = error
            bestPreds = [pred for pred in preds]
            bestK = k
    return bestPreds, minError, bestK

def mlp(x_train, y_train, x_test, y_test):
    #re-scale features to 0->1 range
    mmscaler = MinMaxScaler(feature_range=(0,1))
    x_train = mmscaler.fit_transform(x_train)
    x_test = mmscaler.fit_transform(x_test)

    #train model
    model = neural_network.MLPRegressor(max_iter=10000)

    #optimize params with grid search
    params = {
        "hidden_layer_sizes": [5,10], 
        "activation": ["identity", "logistic", "tanh", "relu"], 
        "solver": ["lbfgs", "sgd", "adam"], 
        "alpha": [0.0005,0.005]
        }
    gsModel = GridSearchCV(estimator=model, param_grid=params)
        
    #fit the model
    # model_fit = model.fit(x_train, np.ravel(y_train))  
    gsModel.fit(x_train, np.ravel(y_train))
    y_pred = gsModel.predict(x_test)
    # print("Params chosen: ", model.get_params())
    error = math.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, error


def runModels(tckr, start, end, yr):
    

    #1 year 
    input = downloadData(ticker=tckr, start=start, end=end)
    x_train, y_train, x_test, y_test, x_train_linreg, x_test_linreg = processInput(input)

    #LinReg
    lrPreds, lrError = linreg(x_train_linreg, y_train, x_test_linreg, y_test)
    lrPreds = reduce(np.append, lrPreds)
    #ARIMA
    arimaPreds, arimaError, bestOrder = arimaGridSearch(x_train_linreg, y_train, x_test, y_test, range(0, 3), range(0, 3), range(0, 3))
    arimaPreds = reduce(np.append, arimaPreds)
    # kNN
    knnPreds, knnError, knnK = knnOptim(x_train_linreg, y_train, x_test, y_test, maxK=10)
    knnPreds = reduce(np.append, knnPreds)
    #mlp
    mlpPreds, mlpError = mlp(x_train_linreg, y_train, x_test, y_test)
    mlpPreds = reduce(np.append, mlpPreds)

    df = pd.DataFrame()
    #add original training data
    # df["Price"] = input["Close"]
    #add model predictions and errors
    df["LinReg"] = lrPreds
    df["LinRegError"] = lrError
    df["ARIMA"] = arimaPreds
    df["ARIMAError"] = arimaError
    df["kNN"] = knnPreds
    df["kNNError"] = knnError
    df["MLP"] = mlpPreds
    df["MLPError"] = mlpError
    
    df.to_csv(""+tckr+"_"+yr+".csv", index=False)




def main():

    start10y = "2010-01-01"
    start5y = "2015-01-01"
    start1y = "2019-01-01"
    end = "2019-12-31"
    

    #TSLA charts
    runModels("TSLA", start1y, end, "1y")
    runModels("TSLA", start5y, end, "5y")
    runModels("TSLA", start10y, end, "10y")

    #MSFT charts
    runModels("MSFT", start1y, end, "1y")
    runModels("MSFT", start5y, end, "5y")
    runModels("MSFT", start10y, end, "10y")

main()