# Object_Detection
TensorFlow Object Detection Demo

# TensorFlow安装 - Anaconda方式
Anaconda方式支持：linux／max ／win ，但是tensorflow for win安装包只能在官方下载，可能会出现连接问题，推荐linux／max

一、Anaconda下载&安装 
- 官网：https://www.continuum.io/downloads/
- 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/  （chmod -x xx.sh    ./xx.sh）

Anaconda 提供一个管理工具  conda ，可以把  conda 看作是  pip +  virtualenv +  PVM (Python Version Manager) + 一些必要的底层库，也就是一个更完整也更大的集成管理工具。
- conda info    //安装成功检查 
- conda env list    //conda虚拟环境list
- conda create —name tensorflow  —clone root    //建立新的conda虚拟环境“tensorflow” 
- source activate tensorflow    //激活“tensorflow”环境

二、TensorFlow安装
- 选择版本-清华镜像：https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/
(tensorflow)$ pip install --ignore-installed —upgrade https://mirrors.tuna.tsinghua.edu.cn/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl
- python
- import tensorflow as tf    //tensorflow安装成功检查
- source deactivate    //关闭“当前tensorflow”环境
