#  _*_ coding: utf-8 _*_
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

dataframe = read_csv(r'E:\data_wdz_mul\data_20min_pre.csv',usecols=[3],engine='python',skipfooter=3)
dataset =dataframe.values
dataset = dataset.astype('float32')

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
# train_size = 1730    #间隔5分钟 前6天训练预测第7天
# train_size = 866  #间隔10分钟 前6天训练预测第7天
train_size = 434  #间隔20分钟 前6天训练预测第7天
#train_size=8641
test_size=len(dataset)-train_size
train,test =dataset[0:train_size,:],dataset[train_size:len(dataset),:]
#use this function to prepare the train and test datasets for modeling
lock_back=1
trainX,trainY=create_dataset(train,lock_back)
testX,testY=create_dataset(test,lock_back)
#reshape input o be [samples,time steps,features]#注意转化数据维数
#把输入重塑成3D格式[样例，时间步，特征]
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX =np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

#建立LSTM模型
model = Sequential()
model.add(LSTM(50,input_shape=(1,lock_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=72,verbose=2)

# 保存模型
# model.save('model_5.h5')    # 间隔为5分钟
# model.save('model_10.h5')    #间隔为10分钟
model.save('model_20.h5')    #间隔为20分钟