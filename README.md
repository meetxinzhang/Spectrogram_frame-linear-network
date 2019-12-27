X 计划 

论文已发表 https://doi.org/10.1016/j.ecoinf.2019.101009 ， 使用本项目中的方法请引用；

本项目受到开源协议保护 Apache License Version 2.0, January 2004 http://www.apache.org/licenses/

使用本代码必须遵循协议规范，注明出处；

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

基于 TensorFlow 

项目内容包括：

一个从 www.xeno-canto.org 下载音频数据的网络爬虫
- 下载列表维护，支持断点续传；
- 捕获网络异常，加入重试机制；
- 可根据重试计数更换代理；
- 模块化设计，可根据文件计数器加入动态代理；

一系列音频数据预处理过程
- 从MP3到梅尔频谱图的转换
- 图像标准化，数据增强
- continuous frame sequence 的构建
- 自动化的 batch 生成器

一个音频识别神经网络
- 以上述3D图像帧序列为输入
- 为音频特殊设计的 Spectrogram-frame linear layer
- 一系列结果分析以及可视化方法


环境配置：
- Tensorflow 1.14以上，需要eager模式支持
- librosa，音频处理
- PIL，图像处理
- scipy，图像处理
- matplotlib，绘图
- sklearn，需要用到一些结果分析函数

