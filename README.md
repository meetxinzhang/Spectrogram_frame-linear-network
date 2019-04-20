X 计划 

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

包括：

一个从 www.xeno-canto.org 下载音频数据的网络爬虫
- 下载列表维护，支持断点续传；
- 捕获网络异常，加入重试机制；
- 可根据重试计数更换代理；
- 模块化设计，可根据文件计数器加入动态代理；

一系列数据预处理过程
- 梅尔频谱图转换
- 图像标准化
- 3D图像序列构建

一个3D CNN + LSTMs 鸟类鸣声识别模型
>3D CNN用于特征提取，多个LSTM并列结构，每个LSTM专注于某一种鸟类的识别，每个LSTM只输出一个概率

这是私有库，保留所有版权，严禁抄袭。

相关比赛

http://otmedia.lirmm.fr/LifeCLEF/BirdCLEF2019/

by Devin Zhang meetdevin.zh@outlook.com
