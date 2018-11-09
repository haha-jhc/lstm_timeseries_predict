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

dataframe = read_csv(r'E:\data_wdz_mul\data_5min_pre.csv',usecols=[3],engine='python',skipfooter=3)
dataset =dataframe.values
dataset = dataset.astype('float32')
if(len(dataframe)==2016):
    train_size=1730
elif(len(dataframe)==1008):
    train_size=866
elif(len(dataframe)==504):
    train_size=433



def create_dataset(dataset,lock_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-lock_back-1):
        a= dataset[i:(i+lock_back),0]
        dataX.append(a)
        dataY.append(dataset[i+lock_back,0])
    return np.array(dataX),np.array(dataY)
def train_test_div(dataset,train_size):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # use this function to prepare the train and test datasets for modeling
    lock_back = 1
    trainX, trainY = create_dataset(train, lock_back)
    testX, testY = create_dataset(test, lock_back)
    return trainX, trainY,testX, testY

trainX, trainY,testX, testY=train_test_div(dataset,train_size)
