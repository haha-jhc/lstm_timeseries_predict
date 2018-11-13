# _*_ coding: utf-8 _*_
"""
LSTM prediction
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#load_dataset
#时间序列做为标签
#dataframe = read_csv(r'E:\data_wdz_mul\data_5min_pre.csv',usecols=[0,2,3],engine='python',index_col=0,skipfooter=3)
dataframe = read_csv(r'E:\data_wdz_mul\data_10min_pre.csv',usecols=[0,2,3],engine='python',index_col=0,skipfooter=3)
#dataframe = read_csv(r'E:\data_wdz_mul\data_20min_pre.csv',usecols=[0,2,3],engine='python',index_col=0,skipfooter=3)
#时间不做标签
# dataframe = read_csv(r'E:\data_wdz_mul\data_5min_pre.csv',usecols=[0,3],engine='python',skipfooter=3)
# dataframe = read_csv(r'E:\data_wdz_mul\data_10min_pre.csv',usecols=[0,3],engine='python',skipfooter=3)
# dataframe = read_csv(r'E:\data_wdz_mul\data_20min_pre.csv',usecols=[0,3],engine='python',skipfooter=3)
values = dataframe.values
#dataset = dataset.astype('float32')
#时间序列做为维度
#dataframe = read_csv(r'E:\data_wdz_mul\data_20min_pre.csv',usecols=[0,3],engine='python')


#convert series to supervised learning
def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars =1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols,names = list(),list()
    #input sequence(t-n,...,t-1)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]
    #forecast sequence (t,t+1,...,t+n)
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names += [('var%d(t)'%(j+1)) for j in range (n_vars)]
        else:
            names += [('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]
    #put it all together
    agg = pd.concat(cols,axis=1)
    agg.columns = names
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)
    return agg
#normalize features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
#frame as supervised learning
reframed = series_to_supervised(scaled,1,1)
reframed.drop(reframed .columns[[2]],axis=1,inplace=True)
print(reframed.head())

#split into input and outputs
values = reframed.values
# train_size = 24*12*6 #5分钟
train_size = 24*6*6  #10分钟
#train_size = 24*3*6  #20分钟
train = values[:train_size,:]
test = values[train_size:,:]
#split into input and outputs
train_X,train_Y = train[:,:-1],train[:,-1]
test_X,test_Y = test[:,:-1],test[:,-1]
#reshape input to be 3D [samples,timesteps,features]
train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

#design network
model = Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae',optimizer='adam')
history = model.fit(train_X,train_Y,epochs=100,batch_size=72,validation_data=(test_X,test_Y),verbose=2,shuffle=False)
# model.save('model_5_mul.h5')
model.save('model_10_mul.h5')
# model.save('model_20_mul.h5')
# model.save('model_5_mult.h5')
# model.save('model_10_mult.h5')
# model.save('model_20_mult.h5')

#plot history
# plt.plot(history.history['loss'],label = 'train')
# plt.plot(history.history['val_loss'],label = 'test')
# plt.legend()
# plt.show()

#model.save('model_5_mul.h5')

yhat=model.predict(test_X)
test_X =test_X.reshape((test_X.shape[0],test_X.shape[2]))
#invert scaling for forecast
inv_yhat = np.concatenate((yhat,test_X[:,1:]),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
#invert scaling for actual
test_Y = test_Y.reshape((len(test_Y),1))
inv_y = np.concatenate((test_Y,test_X[:,1:]),axis=1)
inv_y =scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = math.sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE:%.3f' % rmse)