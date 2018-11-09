#  _*_ coding: utf-8 _*_
"""
LSTM prediction
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize'] = (20,6)
plt.rcParams['font.sans-serif']=['SimHei']
#plt.style.use('ggplot')
#读取数据
#dataframe = read_csv(r'E:\data_all\data_wdz\data_wdz_pre_one.csv',usecols=[3],engine='python',skipfooter=3)
dataframe = read_csv(r'E:\data_wdz_mul\data_5min_pre.csv',usecols=[3],engine='python',skipfooter=3)
dataset =dataframe.values
dataset = dataset.astype('float32')
# convert an array of values into a dataset matrix
def create_dataset(dataset,lock_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-lock_back-1):
        a= dataset[i:(i+lock_back),0]
        dataX.append(a)
        dataY.append(dataset[i+lock_back,0])
    return np.array(dataX),np.array(dataY)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
#split into train and test sets
train_size=1730
#train_size=8641
test_size=len(dataset)-train_size
train,test =dataset[0:train_size,:],dataset[train_size:len(dataset),:]
#use this function to prepare the train and test datasets for modeling
lock_back=1
trainX,trainY=create_dataset(train,lock_back)
testX,testY=create_dataset(test,lock_back)
#reshape input o be [samples,time steps,features]#注意转化数据维数
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX =np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
#建立LSTM模型
model = Sequential()
model.add(LSTM(4,input_shape=(1,lock_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=20,verbose=2)
trainPredict =model.predict(trainX)
testPredict =model.predict(testX)
#数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score:%.6f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score:%.6f RMSE'%(testScore))

train=train[:1729]
#train = train[:8640]
trainPredictPlot =np.empty_like(train)
trainPredictPlot[:,:] =np.nan
trainPredictPlot[lock_back:len(trainPredictPlot)+lock_back,:]=trainPredict

#shift test predictions for plotting
testPredictPlot = np.empty_like(test)
testPredictPlot[:,:] = np.nan
testPredictPlot = testPredict

label=["dataset","testPredict"]
l1,=plt.plot(scaler.inverse_transform(test),color='green')
#l2,=plt.plot(trainPredictPlot,color='blue')
l3,=plt.plot(testPredictPlot,color='orange')
plt.title("间隔5分钟预测图")
plt.legend(label,loc=0,ncol=2)
plt.ylabel("用气量")
plt.show()