# # -*- coding:utf-8 -*-
# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# mu = 100  # 正态分布的均值
# sigma = 15  # 标准差
# x = mu + sigma * np.random.randn(10000)  # 在均值周围产生符合正态分布的x值
#
# num_bins = 50
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
# y = mlab.normpdf(bins, mu, sigma)  # 画一条逼近的曲线
# plt.plot(bins, y, 'r--')
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  # 中文标题 u'xxx'
#
# plt.subplots_adjust(left=0.15)  # 左边距
# plt.show()


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width','petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.head(10))
# descriptions
# print(dataset.describe())
x = dataset.iloc[:,0] #提取第一列的sepal-length变量
mu =np.mean(x) #计算均值
sigma =np.std(x)
mu, sigma

num_bins = 30 #直方图柱子的数量
n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5)
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
y = mlab.normpdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y
plt.plot(bins, y, 'r--') #绘制y的曲线
plt.xlabel('sepal-length') #绘制x轴
plt.ylabel('Probability') #绘制y轴
plt.title(r'Histogram : $\mu=5.8433$,$\sigma=0.8253$')#中文标题 u'xxx'

plt.subplots_adjust(left=0.15)#左边距
plt.show()
