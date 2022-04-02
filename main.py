
import pickle
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error,r2_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM,SimpleRNN
from keras.layers import Dropout
from keras.models import load_model
 
import tensorflow as tf       
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from numpy import mean
from numpy import std 
from math import sqrt 

# ________________________________Librairie sitepackage___________________________________________________________

import streamlit as st
from PIL import Image 
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import datetime
import warnings
warnings.filterwarnings("ignore") 

#____________________________begin dev space______________________________

st.title('    FORECASTING STOCK MARKET   ')
st.write("Used our systeme to predicted your daily return") 
TODAY = date.today().strftime("%Y-%m-%d")
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','AMZN','ABC','ADM','ADP','AEE','BTC','^GSPC')
selected_stock = st.sidebar.selectbox("Select Data to Predict",stocks)
window_selection = st.sidebar.container()
sub_columns = window_selection.columns(2) 
def Business_day( DATE: datetime.date):

         if DATE.weekday() == 1:
             DATE = DATE - datetime.timedelta(days=1)
         return DATE 
YESTERDAY = datetime.date.today() - datetime.timedelta(days =1)
YESTERDAY = Business_day(YESTERDAY)

DEFAULT_START = YESTERDAY - datetime.timedelta(days = 1460)
DEFAULT_START = Business_day(DEFAULT_START) 

START  = sub_columns[0].date_input("FROM", value = DEFAULT_START, max_value = YESTERDAY - datetime.timedelta(days = 1))
END = sub_columns[1].date_input("TO", value = YESTERDAY, max_value = YESTERDAY, min_value = START)
START = Business_day(START)
END = Business_day(END)
#_____________________________________select  your model name and load data____________________________________________________________

model_name = st.sidebar.selectbox(
    'Selected Model',
    ('RNN', 'KNN', 'LSTM','DT','MLP','SVR')
)
image = Image.open('marché.jpg')
st.image(image, caption='Marché Boursier')

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Visualisation des données Brutes')
st.write(data)

def plot_raw_data():
	fig = go.Figure()
     
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Evolution Actuelle', xaxis_rangeslider_visible=True,width=850,height=600,autosize=False)
	st.plotly_chart(fig)
plot_raw_data()
 

#____________________________MODEL SVR_________________________________________________________________
tab = []
with open('style.css') as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
btn_train = st.sidebar.button("Make_Train")
day_select = st.sidebar.select_slider('select your day to predict',range(1,30))
btn_predict = st.sidebar.button("Make_Predict")

if btn_train :
    if model_name == 'SVR':
        def SVR_Model(data_name) :
        
            data_close = np.array(data['Close'])
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))
        
            step_size=1
            train_len = int(len(data_close) * 0.75)
            test_len = len(data_close) - train_len
            train_Data, test_Data = data_close[0:train_len,:], data_close[train_len:len(data_close),:]
        
            def new_dataset(dataset):
                x_train, y_train = [], [] 
                for i in range(len(dataset)-step_size-1):
                    a = dataset[i:(i+step_size), 0]
                    x_train.append(a) 
                    y_train.append(dataset[i + step_size, 0])
                return np.array(x_train), np.array(y_train)
        
            x_train, y_train = new_dataset(train_Data )
            x_test, y_test = new_dataset(test_Data )
        
            gsc_rbf = GridSearchCV(
                estimator=SVR(kernel='rbf'),
                param_grid={
                    'C': [0.01, 1, 100,500, 1000],
                    'coef0' : [0.01,0.5,1],
                    'epsilon': [0.0001, 0.0005, 0.001],
                    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            st.write("###                      SVR_Model ###")
            grid_rbf = gsc_rbf.fit(x_train, y_train)
            best_params = grid_rbf.best_params_
            model_svr = SVR(kernel= 'rbf',C=best_params['C'],coef0=best_params['coef0'],epsilon=best_params['epsilon'],gamma=best_params['gamma'])
        
            file = open("SVR_LOO.txt","w+")
            i=0
            scores_MAE=[]
            scores_RMSE=[]
    
            leaveout = LeaveOneOut() 
    
            for train_index, test_index in leaveout.split(x_train): 
                X_train, X_test = x_train[train_index], x_train[test_index]
                Y_train, Y_test = y_train[train_index], y_train[test_index]
            
                model_svr.fit(X_train, Y_train)
                Y_pred = model_svr.predict(X_test)
                
                MAE = mean_absolute_error(Y_test, Y_pred)
                RMSE = sqrt(mean_squared_error(Y_test,Y_pred))
    
                scores_MAE.append(MAE)
                scores_RMSE.append(RMSE) 
            
                s = str(i)+","+str(float(Y_test))+","+str(float(Y_pred))+","+str(MAE)+","+str(RMSE)+"\n"
                L=file.readlines()
                L.insert(i,s)
                file.writelines(L)
                i=i+1
        
            file.close()
            
            filename = 'model_svr.sav'
            pickle.dump(model_svr, open(filename, 'wb'))
            train_Predict = model_svr.predict(x_train.reshape(-1,1))
            test_Predict = model_svr.predict(x_test.reshape(-1,1))
    
            MAE = np.sqrt(mean_absolute_error(y_train,train_Predict))
            RMSE =  np.sqrt(mean_squared_error(y_test,test_Predict))
            score = model_svr.score(x_test,y_test)
            tab.append(score)
            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
            data_close=scaler.inverse_transform(data_close)
            train_Predict= scaler.inverse_transform(train_Predict.reshape(-1,1))
            test_Predict= scaler.inverse_transform(test_Predict.reshape(-1,1))
            
            fig = plt.figure(figsize=(10,5))  
            plt.plot(data['Date'],data_close, 'b', label = 'original dataset')
            plt.plot(data.iloc[:len(y_train),0] ,train_Predict, 'r', label = 'training set')
            plt.plot(data.iloc[len(data)-len(y_test):,0] ,test_Predict, 'g', label = 'testing set') 
            plt.legend(loc = 'upper left')
            plt.xlabel('Time in Date format')
            plt.ylabel('Close values Stocks') 
            plt.show()
            st.pyplot(fig)

            return MAE,RMSE,score

        MAE,RMSE,score = SVR_Model(data)

        with open('style.css') as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("RMSE")
            st.write( RMSE)
        with col2:
            st.header("MAE")
            st.write( MAE)

        with col3:
            st.header("SCORE")
            st.write( score)

#____________________________________MODEL LSTM_____________________________________________________________

    elif model_name == 'MLP':
        def MLP_Model(data_name) :
        
            data_close = np.array(data['Close'])
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))
        
            step_size=1
            train_len = int(len(data_close) * 0.75)
            test_len = len(data_close) - train_len
            train_Data, test_Data = data_close[0:train_len,:], data_close[train_len:len(data_close),:]
        
            def new_dataset(dataset):
                x_train, y_train = [], []
                for i in range(len(dataset)-step_size-1):
                    a = dataset[i:(i+step_size), 0]
                    x_train.append(a) 
                    y_train.append(dataset[i + step_size, 0])
                return np.array(x_train), np.array(y_train)
        
            x_train, y_train = new_dataset(train_Data )
            x_test, y_test = new_dataset(test_Data )

            gsc_MLP = GridSearchCV(
                estimator=MLPRegressor(),
                param_grid={
                    'hidden_layer_sizes': [(10,30,10),(20,20,20),(20,50,30),(50,50,50)],
                    'activation': ['logistic','sigmoid','linear','softmax','tanh', 'relu'],
                    'solver': ['sgd', 'adam','lbfgs'],
                    'alpha': [0.0001,0.001,0.005,0.01,0.05,0.05,0.1],
                    'learning_rate': ['constant','invscaling','adaptive']
                },
                cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            st.write("###                      MLPRegressor ###")
            grid_MLP = gsc_MLP.fit(x_train, y_train)
            best_params = grid_MLP.best_params_
            grid_MLP.best_params_
            grid_MLP.best_score_

            
            model_MLP = MLPRegressor(hidden_layer_sizes= best_params['hidden_layer_sizes'],activation= best_params['activation'],
                        solver=best_params['solver'],alpha=best_params['alpha'],
                        learning_rate=best_params['learning_rate'])
            

            file = open("MLP_LOO.txt","w+")
            i=0
            scores_MAE=[]
            scores_RMSE=[]
    
            leaveout = LeaveOneOut() 
        
            for train_index, test_index in leaveout.split(x_train): 
                X_train, X_test = x_train[train_index], x_train[test_index]
                Y_train, Y_test = y_train[train_index], y_train[test_index]
            
                model_MLP.fit(X_train, Y_train)
                Y_pred = model_MLP.predict(X_test)
        
                MAE = mean_absolute_error(Y_test, Y_pred)
                RMSE = sqrt(mean_squared_error(Y_test,Y_pred))
      
                scores_MAE.append(MAE)
                scores_RMSE.append(RMSE) 
            
                s = str(i)+","+str(float(Y_test))+","+str(float(Y_pred))+","+str(MAE)+","+str(RMSE)+"\n"
                L=file.readlines()
                L.insert(i,s)
                file.writelines(L)
                i=i+1
        
            file.close()
            
            filename = 'model_mlp.sav'
            pickle.dump(model_MLP, open(filename, 'wb'))
            train_Predict = model_MLP.predict(x_train.reshape(-1,1))
            test_Predict = model_MLP.predict(x_test.reshape(-1,1))
            MAE= np.sqrt(mean_absolute_error(y_test,test_Predict))
            RMSE =  np.sqrt(mean_squared_error(y_test,test_Predict))
            score = model_MLP.score(x_test,y_test)

            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
            tab.append(score)
            data_close=scaler.inverse_transform(data_close)
            train_Predict= scaler.inverse_transform(train_Predict.reshape(-1,1))
            test_Predict= scaler.inverse_transform(test_Predict.reshape(-1,1))
            fig = plt.figure(figsize=(10,5))
            plt.plot(data['Date'],data_close, 'b', label = 'original dataset')
            plt.plot(data.iloc[:len(y_train),0] ,train_Predict, 'r', label = 'training set')
            plt.plot(data.iloc[len(data)-len(y_test):,0] ,test_Predict, 'g', label = 'testing set') 
            plt.legend(loc = 'upper left')
            plt.xlabel('Time in date format')
            plt.ylabel('Close values Stocks') 
            plt.show()
            st.pyplot(fig)

            return RMSE,MAE,score
        
        RMSE,MAE,score = MLP_Model(data)

        with open('style.css') as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("RMSE")
            st.write( RMSE)
        with col2:
            st.header("MAE")
            st.write( MAE)

        with col3:
            st.header("SCORE")
            st.write( score)
        
    #___________________________________MODEL KNN_________________________________________________

    elif model_name == 'KNN':

        def KNN_Model(data_name) :
        
            data_close = np.array(data['Close'])
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))
        
            step_size=1
            train_len = int(len(data_close) * 0.75)
            test_len = len(data_close) - train_len
            train_Data, test_Data = data_close[0:train_len,:], data_close[train_len:len(data_close),:]
        
            def new_dataset(dataset):
                x_train, y_train = [], []
                for i in range(len(dataset)-step_size-1):
                    a = dataset[i:(i+step_size), 0]
                    x_train.append(a) 
                    y_train.append(dataset[i + step_size, 0])
                return np.array(x_train), np.array(y_train)
        
            x_train, y_train = new_dataset(train_Data )
            x_test, y_test = new_dataset(test_Data )

            gsc_knn = GridSearchCV(
                estimator=KNeighborsRegressor(),
                param_grid={
                    'n_neighbors': [3,5,7,9,11,15,21,29,41,55],
                    'weights' : ['uniform', 'distance'],
                    'metric': ['euclidean','manhattan','minkowski'],
                    'algorithm': ['auto', 'ball_tree','kd_tree', 'brute']
                },
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            st.write("###                      KNeighborsRegressor ###")
            grid_knn = gsc_knn.fit(x_train, y_train)
            best_params = grid_knn.best_params_

            
            model_KNN = KNeighborsRegressor(algorithm=best_params['algorithm'],
                                metric=best_params['metric'],n_neighbors=best_params['n_neighbors'],
                                    weights=best_params['weights'])    
        
            
            file = open("KNN_LOO.txt","w+")
            i=0
            scores_MAE=[]
            scores_RMSE=[]
    
            leaveout = LeaveOneOut() 
            for train_index, test_index in leaveout.split(x_train):
                    X_train, X_test = x_train[train_index], x_train[test_index]
                    Y_train, Y_test = y_train[train_index], y_train[test_index]
            
                    model_KNN.fit(X_train, Y_train)
                    Y_pred = model_KNN.predict(X_test)
        
                    MAE = mean_absolute_error(Y_test, Y_pred)
                    RMSE = sqrt(mean_squared_error(Y_test,Y_pred))
    
                    scores_MAE.append(MAE)
                    scores_RMSE.append(RMSE) 
            
                    s = str(i)+","+str(float(Y_test))+","+str(float(Y_pred))+","+str(MAE)+","+str(RMSE)+"\n"
                    L=file.readlines()
                    L.insert(i,s)
                    file.writelines(L)
                    i=i+1 
            file.close()
            
            filename = 'model_knn.sav'
            pickle.dump(model_KNN, open(filename, 'wb'))
            train_Predict = model_KNN.predict(x_train.reshape(-1,1))
            test_Predict = model_KNN.predict(x_test.reshape(-1,1))
            MAE = np.sqrt(mean_absolute_error(y_test,test_Predict))
            RMSE =  np.sqrt(mean_squared_error(y_test,test_Predict))
            score = model_KNN.score(x_test,y_test)

            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
            tab.append(score) 
            data_close=scaler.inverse_transform(data_close)
            train_Predict= scaler.inverse_transform(train_Predict.reshape(-1,1))
            test_Predict= scaler.inverse_transform(test_Predict.reshape(-1,1))

            fig = plt.figure(figsize=(10,5))
            plt.plot(data['Date'],data_close, 'b', label = 'original dataset')
            plt.plot(data.iloc[:len(y_train),0] ,train_Predict, 'r', label = 'training set')
            plt.plot(data.iloc[len(data)-len(y_test):,0] ,test_Predict, 'g', label = 'testing set') 
            plt.legend(loc = 'upper left')
            plt.xlabel('Time Date format')
            plt.ylabel('Close Values Stocks') 
            plt.show()
            st.pyplot(fig)

            return RMSE,MAE,score
        
        RMSE,MAE,score = KNN_Model(data)

        with open('style.css') as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("RMSE")
            st.write( RMSE)
        with col2:
            st.header("MAE")
            st.write( MAE)

        with col3:
            st.header("SCORE")
            st.write( score)

    #____________________________________MODEL DT__________________________________________________________

    elif model_name == 'DT':
        def DT_Model(data) :
        
            data_close = np.array(data['Close'])
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))
        
            step_size=1
            train_len = int(len(data_close) * 0.75)
            test_len = len(data_close) - train_len
            train_Data, test_Data = data_close[0:train_len,:], data_close[train_len:len(data_close),:]
        
            def new_dataset(dataset):
                x_train, y_train = [], []
                for i in range(len(dataset)-step_size-1):
                    a = dataset[i:(i+step_size), 0]
                    x_train.append(a) 
                    y_train.append(dataset[i + step_size, 0])
                return np.array(x_train), np.array(y_train)
        
            x_train, y_train = new_dataset(train_Data )
            x_test, y_test = new_dataset(test_Data )

            gsc_DT = GridSearchCV(
            estimator=DecisionTreeRegressor(),
            param_grid={
                "criterion": ["mse", "friedman_mse", "mae", "poisson"],
                "splitter":["best", "random"],
                "min_samples_split": [1,2,5,10,15,20,30,40,55,67,90],
                "max_depth": [2, 6,8,11,14],
                "min_samples_leaf": [5,10,20, 40, 100],
                "max_leaf_nodes": [5, 20,50, 100]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        
            st.write("###                      DecisionTreeRegressor ###")
            grid_DT = gsc_DT.fit(x_train, y_train)
            best_params = grid_DT.best_params_
            grid_DT.best_params_
            grid_DT.best_score_
            
            model_DT = DecisionTreeRegressor(criterion=best_params['criterion'],splitter=best_params['splitter'],
                                min_samples_split=best_params['min_samples_split'],max_depth=best_params['max_depth'],
                                min_samples_leaf=best_params['min_samples_leaf'],
                                max_leaf_nodes=best_params['max_leaf_nodes'])
            
            file = open("DT_LOO.txt","w+")
            i=0
            scores_MAE=[]
            scores_RMSE=[]
    
            leaveout = LeaveOneOut() 
        
            for train_index, test_index in leaveout.split(x_train): 
                X_train, X_test = x_train[train_index], x_train[test_index]
                Y_train, Y_test = y_train[train_index], y_train[test_index]
            
                model_DT.fit(X_train, Y_train)
                Y_pred = model_DT.predict(X_test)
        
                MAE = mean_absolute_error(Y_test, Y_pred)
                RMSE = sqrt(mean_squared_error(Y_test,Y_pred))
    
                scores_MAE.append(MAE)
                scores_RMSE.append(RMSE) 
            
                s = str(i)+","+str(float(Y_test))+","+str(float(Y_pred))+","+str(MAE)+","+str(RMSE)+"\n"
                L=file.readlines()
                L.insert(i,s)
                file.writelines(L)
                i=i+1
            
            file.close()
            
            filename = 'model_dt.sav'
            pickle.dump(model_DT, open(filename, 'wb'))
            train_Predict = model_DT.predict(x_train.reshape(-1,1))
            test_Predict = model_DT.predict(x_test.reshape(-1,1))
            MAE = np.sqrt(mean_absolute_error(y_test,test_Predict))
            RMSE =  np.sqrt(mean_squared_error(y_test,test_Predict))
            score = model_DT.score(x_test,y_test)

            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
            tab.append(score)
            data_close=scaler.inverse_transform(data_close)
            train_Predict= scaler.inverse_transform(train_Predict.reshape(-1,1))
            test_Predict= scaler.inverse_transform(test_Predict.reshape(-1,1))

            fig = plt.figure(figsize=(10,5))
            plt.plot(data['Date'],data_close, 'b', label = 'original dataset')
            plt.plot(data.iloc[:len(y_train),0] ,train_Predict, 'r', label = 'training set')
            plt.plot(data.iloc[len(data)-len(y_test):,0] ,test_Predict, 'g', label = 'testing set') 
            plt.legend(loc = 'upper left')
            plt.xlabel('Time in Date format')
            plt.ylabel('Close Values Stocks') 
            plt.show()
            st.pyplot(fig)

            return RMSE,MAE,score,model_DT
        
        RMSE,MAE,score,model_DT = DT_Model(data)

        with open('style.css') as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("RMSE")
            st.write( RMSE)
        with col2:
            st.header("MAE")
            st.write( MAE)

        with col3:
            st.header("SCORE")
            st.write( score)
    #_____________________________________MODEL LSTM_____________________________________________________

    elif model_name == 'LSTM':
        
            data_close=data.reset_index()['Close']
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))

            training_size=int(len(data_close)*0.70)
            test_size=len(data_close)-training_size
            train_data,test_data=data_close[0:training_size,:],data_close[training_size:len(data_close),:1]

            def create_dataset(dataset, time_step=100):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]  
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 100
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, ytest = create_dataset(test_data, time_step)

            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            model_L=Sequential()
            model_L.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            model_L.add(LSTM(50,return_sequences=True)) 
            model_L.add(LSTM(50))
            model_L.add(Dense(1))
            model_L.compile(loss='mean_squared_error',optimizer='adam')
            
            model_L.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
            model_L.save("moldel_l.h5")
            train_predict=model_L.predict(X_train)
            test_predict=model_L.predict(X_test)
            
            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)
            y_train=scaler.inverse_transform(y_train.reshape(-1,1))

            MAE =np.sqrt(mean_absolute_error(ytest,model_L.predict(X_test)))
            RMSE =np.sqrt(mean_squared_error(ytest,model_L.predict(X_test)))
            score = model_L.evaluate(X_test, ytest, verbose=1)
            
            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
            tab.append(score) 

            look_back=100
            trainPredictPlot = np.empty_like(data_close)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
            
            testPredictPlot = np.empty_like(data_close)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict)+(look_back*2)+1:len(data_close)-1, :] = test_predict
            
            st.write("##                      Long_Short_Term_Memory: LSTM ##")
            fig = plt.figure(figsize=(10,5))
            plt.plot(scaler.inverse_transform(data_close),'b',label='original data')
            plt.plot(trainPredictPlot,'r',label='Training set')
            plt.plot(testPredictPlot,'g',label='Testing set')
            plt.legend(loc = 'upper left')
            plt.xlabel('Time in days')
            plt.ylabel('Close Values Stocks') 
            plt.show()
            st.pyplot(fig)
            
            with open('style.css') as f:
               st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            col1, col2,col3= st.columns(3)

            with col1:
                st.header("RMSE")
                st.write( RMSE)
            with col2:
                st.header("MAE")
                st.write( MAE)
            # with col3:
            #     st.header("score")
            #     st.write( MAE)
 
 # _________________________MODEL RNN_____________________________________________________

    elif model_name == 'RNN':
        
            data_close=data.reset_index()['Close']
            scaler=MinMaxScaler(feature_range=(0,1))
            data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))

            training_size=int(len(data_close)*0.70)
            test_size=len(data_close)-training_size
            train_data,test_data=data_close[0:training_size,:],data_close[training_size:len(data_close),:1]

            def create_dataset(dataset, time_step=100):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]  
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 100
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, ytest = create_dataset(test_data, time_step)

            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            model_R=Sequential()
            model_R.add(SimpleRNN(20,return_sequences=True,input_shape=(X_train.shape[1],1)))
            model_R.add(SimpleRNN(20,return_sequences=True)) 
            model_R.add(SimpleRNN(20))
            model_R.add(Dense(1))
            model_R.compile(loss='mean_squared_error',optimizer='adam')

            model_R.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
            model_R.save('model_r.h5')
            train_predict=model_R.predict(X_train)
            test_predict=model_R.predict(X_test)

            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)
            y_train=scaler.inverse_transform(y_train.reshape(-1,1))

            MAE =np.sqrt(mean_absolute_error(ytest,model_R.predict(X_test)))
            RMSE =np.sqrt(mean_squared_error(ytest,model_R.predict(X_test)))
            score = model_R.evaluate(X_test, ytest, verbose=1)
            
            MAE = round(MAE,10)
            RMSE = round(RMSE,10)
            score = round(score,4)
             

            look_back=100
            trainPredictPlot = np.empty_like(data_close)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
            
            testPredictPlot = np.empty_like(data_close)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict)+(look_back*2)+1:len(data_close)-1, :] = test_predict
            
            st.write("###                      SimpleRNN Model: RNN ###")
            fig = plt.figure(figsize=(10,5))
            plt.plot(scaler.inverse_transform(data_close),'b',label='original data')
            plt.plot(trainPredictPlot,'r',label='Training set')
            plt.plot(testPredictPlot,'g',label='Testing set')
            plt.legend(loc = 'upper left')
            plt.xlabel('Time in days')
            plt.ylabel('Close Values Stocks') 
            plt.show()
            st.pyplot(fig)

            with open('style.css') as f:
               st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            col1, col2,col3= st.columns(3)

            with col1:
                st.header("RMSE")
                st.write( RMSE)
            with col2:
                st.header("MAE")
                st.write( MAE) 
            # with col3:
            #     st.header("score")
            #     st.write( score)
 
            

 #_____________________________BUILD METHODE TO MAKE NEXTS DAYS PREDICT ____________________________________________

elif btn_predict:
    if model_name =='RNN':
        model=load_model('model_r.h5')
        data_close=data.reset_index()['Close']
        data_High= data.reset_index()['High']
        data_Low=data.reset_index()['Low']
        scaler=MinMaxScaler(feature_range=(0,1))
        data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))

        training_size=int(len(data_close)*0.70)
        test_size=len(data_close)-training_size
        train_data,test_data=data_close[0:training_size,:],data_close[training_size:len(data_close),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_close.tolist()
        ds_new.extend(lst_output)
        final_graph = scaler.inverse_transform(ds_new).tolist()

        data_High=scaler.fit_transform(np.array(data_High).reshape(-1,1))
        training_size=int(len(data_High)*0.70)
        test_size=len(data_High)-training_size
        train_data,test_data=data_High[0:training_size,:],data_High[training_size:len(data_High),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_High.tolist()
        ds_new.extend(lst_output)
        final_graph2 = scaler.inverse_transform(ds_new).tolist()

        data_Low=scaler.fit_transform(np.array(data_Low).reshape(-1,1))
        training_size=int(len(data_Low)*0.70)
        test_size=len(data_Low)-training_size
        train_data,test_data=data_Low[0:training_size,:],data_Low[training_size:len(data_Low),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_Low.tolist()
        ds_new.extend(lst_output)
        final_graph3 = scaler.inverse_transform(ds_new).tolist()

        fig = plt.figure(figsize=(10,5))
        plt.plot(final_graph2)
        plt.plot(final_graph)
        plt.plot(final_graph3) 
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'next High value : {0}'.format(round(float(*final_graph2[len(final_graph2)-1]),2)))
        plt.axhline(y=final_graph[len(final_graph)-1], color = 'g', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
        plt.axhline(y=final_graph3[len(final_graph3)-1], color = 'b', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*final_graph3[len(final_graph3)-1]),2)))
        plt.legend()
        st.write("show your prediction with LSTM Model")
        st.pyplot(fig)
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*final_graph2[len(final_graph2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*final_graph[len(final_graph)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*final_graph3[len(final_graph3)-1]),2))
          
        nbr=len(final_graph)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(final_graph2)
        with col5:
            st.write("# Close value #")        
            st.write(final_graph)
        with col6:   
            st.write("# Low value #")        
            st.write(final_graph3)

    elif model_name =='LSTM':
        model=load_model('moldel_l.h5')
        data_close=data.reset_index()['Close']
        data_High= data.reset_index()['High']
        data_Low=data.reset_index()['Low']
        scaler=MinMaxScaler(feature_range=(0,1))
        data_close=scaler.fit_transform(np.array(data_close).reshape(-1,1))

        training_size=int(len(data_close)*0.70)
        test_size=len(data_close)-training_size
        train_data,test_data=data_close[0:training_size,:],data_close[training_size:len(data_close),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_close.tolist()
        ds_new.extend(lst_output)
        final_graph = scaler.inverse_transform(ds_new).tolist()

        data_High=scaler.fit_transform(np.array(data_High).reshape(-1,1))
        training_size=int(len(data_High)*0.70)
        test_size=len(data_High)-training_size
        train_data,test_data=data_High[0:training_size,:],data_High[training_size:len(data_High),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_High.tolist()
        ds_new.extend(lst_output)
        final_graph2 = scaler.inverse_transform(ds_new).tolist()

        data_Low=scaler.fit_transform(np.array(data_Low).reshape(-1,1))
        training_size=int(len(data_Low)*0.70)
        test_size=len(data_Low)-training_size
        train_data,test_data=data_Low[0:training_size,:],data_Low[training_size:len(data_Low),:1]
        
        fut_inp = test_data[len(test_data)-100:]
        fut_inp = fut_inp.reshape(1,-1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        nbr_days=day_select
        while(i<nbr_days):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
            
        print(lst_output)

        ds_new = data_Low.tolist()
        ds_new.extend(lst_output)
        final_graph3 = scaler.inverse_transform(ds_new).tolist()

        fig = plt.figure(figsize=(10,5))
        plt.plot(final_graph2)
        plt.plot(final_graph)
        plt.plot(final_graph3) 
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'next High value : {0}'.format(round(float(*final_graph2[len(final_graph2)-1]),2)))
        plt.axhline(y=final_graph[len(final_graph)-1], color = 'g', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
        plt.axhline(y=final_graph3[len(final_graph3)-1], color = 'b', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*final_graph3[len(final_graph3)-1]),2)))
        plt.legend()
        st.write("show your prediction with LSTM Model")
        st.pyplot(fig)
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*final_graph2[len(final_graph2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*final_graph[len(final_graph)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*final_graph3[len(final_graph3)-1]),2))
        
        nbr=len(final_graph)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(final_graph2)
        with col5:
            st.write("# Close value #")        
            st.write(final_graph)
        with col6:   
            st.write("# Low value #")        
            st.write(final_graph3)
        
        
    elif model_name=='SVR':

        model=pickle.load(open('model_svr.sav', 'rb'))
        df=data[['Close']]
        df2=data[['Close']]
        df3 = data[['High']]
        df4=data[['High']]
        df5 = data[['Low']]
        df6=data[['Low']]


        df3['Prediction'] = df3[['High']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast2 = np.array(df3.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast2)
        normalize.transform(forecast2) 
        predict2 = model.predict(normalize.transform(forecast2))
        pred_value2=normalize.inverse_transform(predict2.reshape(-1,1))
        data_final2 = np.vstack((df4,pred_value2))

        df['Prediction'] = df[['Close']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast = np.array(df.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast)
        normalize.transform(forecast)
        predict = model.predict(normalize.transform(forecast))
        pred_value=normalize.inverse_transform(predict.reshape(-1,1))
        data_final = np.vstack((df2,pred_value))

        df5['Prediction'] = df5[['Low']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast3 = np.array(df5.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast3)
        normalize.transform(forecast3) 
        predict3 = model.predict(normalize.transform(forecast3))
        pred_value3=normalize.inverse_transform(predict3.reshape(-1,1))
        data_final3 = np.vstack((df6,pred_value3))

        fig = plt.figure(figsize=(10,5))
        plt.plot(data_final2)
        plt.plot(data_final)
        plt.plot(data_final3)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=data_final[len(data_final)-1], color = 'b', linestyle = ':', label = 'next High value : {0}'.format(round(float(*data_final2[len(data_final2)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'red', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*data_final[len(data_final)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'g', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*data_final3[len(data_final3)-1]),2)))
        plt.legend()
        st.write("show your prediction with KNN Model")
        st.pyplot(fig)

        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*data_final2[len(data_final2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*data_final[len(data_final)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*data_final3[len(data_final3)-1]),2))
        
        nbr=len(data_final)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(data_final2)
        with col5:
            st.write("# Close value #")        
            st.write(data_final)
        with col6:   
            st.write("# Low value #")        
            st.write(data_final3)
        
    elif model_name=='MLP':
        model=pickle.load(open('model_mlp.sav', 'rb'))
        df=data[['Close']]
        df2=data[['Close']]
        df3 = data[['High']]
        df4=data[['High']]
        df5 = data[['Low']]
        df6=data[['Low']]

        df3['Prediction'] = df3[['High']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast2 = np.array(df3.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast2)
        normalize.transform(forecast2) 
        predict2 = model.predict(normalize.transform(forecast2))
        pred_value2=normalize.inverse_transform(predict2.reshape(-1,1))
        data_final2 = np.vstack((df4,pred_value2))

        df['Prediction'] = df[['Close']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast = np.array(df.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast)
        normalize.transform(forecast)
        predict = model.predict(normalize.transform(forecast))
        pred_value=normalize.inverse_transform(predict.reshape(-1,1))
        data_final = np.vstack((df2,pred_value))

        
        df5['Prediction'] = df5[['Low']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast3 = np.array(df5.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast3)
        normalize.transform(forecast3) 
        predict3 = model.predict(normalize.transform(forecast3))
        pred_value3=normalize.inverse_transform(predict3.reshape(-1,1))
        data_final3 = np.vstack((df6,pred_value3))

        fig = plt.figure(figsize=(10,5))
        plt.plot(data_final2)
        plt.plot(data_final)
        plt.plot(data_final3)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=data_final[len(data_final)-1], color = 'b', linestyle = ':', label = 'next High value : {0}'.format(round(float(*data_final2[len(data_final2)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'red', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*data_final[len(data_final)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'g', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*data_final3[len(data_final3)-1]),2)))
        plt.legend()
        st.write("show your prediction with KNN Model")
        st.pyplot(fig)

        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*data_final2[len(data_final2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*data_final[len(data_final)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*data_final3[len(data_final3)-1]),2))

        nbr=len(data_final)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(data_final2)
        with col5:
            st.write("# Close value #")        
            st.write(data_final)
        with col6:   
            st.write("# Low value #")        
            st.write(data_final3)

    elif model_name=="DT":
        model=pickle.load(open('model_dt.sav', 'rb'))
        df=data[['Close']]
        df2=data[['Close']]
        df3 = data[['High']]
        df4=data[['High']]
        df5 = data[['Low']]
        df6=data[['Low']]


        df3['Prediction'] = df3[['High']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast2 = np.array(df3.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast2)
        normalize.transform(forecast2) 
        predict2 = model.predict(normalize.transform(forecast2))
        pred_value2=normalize.inverse_transform(predict2.reshape(-1,1))
        data_final2 = np.vstack((df4,pred_value2))

        df['Prediction'] = df[['Close']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast = np.array(df.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast)
        normalize.transform(forecast)
        predict = model.predict(normalize.transform(forecast))
        pred_value=normalize.inverse_transform(predict.reshape(-1,1))
        data_final = np.vstack((df2,pred_value))

        
        df5['Prediction'] = df5[['Low']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast3 = np.array(df5.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast3)
        normalize.transform(forecast3) 
        predict3 = model.predict(normalize.transform(forecast3))
        pred_value3=normalize.inverse_transform(predict3.reshape(-1,1))
        data_final3 = np.vstack((df6,pred_value3))

        fig = plt.figure(figsize=(10,5))
        plt.plot(data_final2)
        plt.plot(data_final)
        plt.plot(data_final3)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=data_final[len(data_final)-1], color = 'b', linestyle = ':', label = 'next High value : {0}'.format(round(float(*data_final2[len(data_final2)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'red', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*data_final[len(data_final)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'g', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*data_final3[len(data_final3)-1]),2)))
        plt.legend()
        st.write("show your prediction with KNN Model")
        st.pyplot(fig)

        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*data_final2[len(data_final2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*data_final[len(data_final)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*data_final3[len(data_final3)-1]),2))
        
        nbr=len(data_final)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(data_final2)
        with col5:
            st.write("# Close value #")        
            st.write(data_final)
        with col6:   
            st.write("# Low value #")        
            st.write(data_final3)

    elif model_name=="KNN":
        model=pickle.load(open('model_knn.sav', 'rb'))
        df=data[['Close']]
        df2=data[['Close']]
        df3 = data[['High']]
        df4=data[['High']]
        df5 = data[['Low']]
        df6=data[['Low']]


        df3['Prediction'] = df3[['High']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast2 = np.array(df3.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast2)
        normalize.transform(forecast2) 
        predict2 = model.predict(normalize.transform(forecast2))
        pred_value2=normalize.inverse_transform(predict2.reshape(-1,1))
        data_final2 = np.vstack((df4,pred_value2))

        df['Prediction'] = df[['Close']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast = np.array(df.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast)
        normalize.transform(forecast)
        predict = model.predict(normalize.transform(forecast))
        pred_value=normalize.inverse_transform(predict.reshape(-1,1))
        data_final = np.vstack((df2,pred_value))

        
        df5['Prediction'] = df5[['Low']].shift(-day_select)
        normalize=MinMaxScaler(feature_range=(0,1))
        forecast3 = np.array(df5.drop(['Prediction'],1))[-day_select:]
        normalize.fit(forecast3)
        normalize.transform(forecast3) 
        predict3 = model.predict(normalize.transform(forecast3))
        pred_value3=normalize.inverse_transform(predict3.reshape(-1,1))
        data_final3 = np.vstack((df6,pred_value3))

        fig = plt.figure(figsize=(10,5))
        plt.plot(data_final2)
        plt.plot(data_final)
        plt.plot(data_final3)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0}, next predicted  ".format(selected_stock))
        plt.axhline(y=data_final[len(data_final)-1], color = 'b', linestyle = ':', label = 'next High value : {0}'.format(round(float(*data_final2[len(data_final2)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'red', linestyle = ':', label = 'next Close value : {0}'.format(round(float(*data_final[len(data_final)-1]),2)))
        plt.axhline(y=data_final[len(data_final)-1], color = 'g', linestyle = ':', label = 'next Low value : {0}'.format(round(float(*data_final3[len(data_final3)-1]),2)))
        plt.legend()
        st.write("show your prediction with KNN Model")
        st.pyplot(fig)

        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Max")
            st.write(round(float(*data_final2[len(data_final2)-1]),2))
        with col2:
            st.header("Close ")
            st.write(round(float(*data_final[len(data_final)-1]),2)) 
        with col3:
            st.header(" Min")
            st.write(round(float(*data_final3[len(data_final3)-1]),2))
        
        nbr=len(data_final)-day_select
        st.write("# see your prediction results per day from the line {} #".format(nbr))
        col4, col5, col6 = st.columns(3)
        with col4:
           st.write("# Max value #")
           st.write(data_final2)
        with col5:
            st.write("# Close value #")        
            st.write(data_final)
        with col6:   
            st.write("# Low value #")        
            st.write(data_final3)
 


    
    