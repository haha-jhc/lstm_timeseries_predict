import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
data=read_csv(r'E:\data_wdz_mul\data_5min_pre.csv',encoding='gbk',usecols=[0,1,2,3],parse_dates=['GathDt'])
#data=read_csv(r'E:\data_all\data_wdz\data_wdz_pre.csv',usecols=[2],encoding='gbk',parse_dates=['GathDt'])
table = pd.pivot_table(data,index=['GathDt'],values=['YQL'])
table = table[0:288]
fig=plt.figure(figsize=(30,10),dpi=100)#设置图线大小
ax = fig.add_subplot(111)#出图个数
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))#设置图线的时间尺度
plt.xticks(pd.date_range(table.index[0],table.index[-1],freq='H'),rotation=45)#按小时标记时间坐标
# ax.plot(table.index,table['用气量'],color='r')#设置取曲线样式
plt.title('2018年2月1日数据-5分钟(5平滑)')
ax.plot(table.index,table['YQL'].rolling(5).mean(),color='b')
plt.show()