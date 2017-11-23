# Object_Detection
TensorFlow Object Detection Demo

# 一、TensorFlow安装 - Anaconda方式
Anaconda方式支持：linux／max ／win ，但是tensorflow for win安装包只能在官方下载，可能会出现连接问题，推荐linux／max

1、Anaconda下载&安装 
- 官网：https://www.continuum.io/downloads/
- 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/  （chmod -x xx.sh    ./xx.sh）

Anaconda 提供一个管理工具  conda ，可以把  conda 看作是  pip +  virtualenv +  PVM (Python Version Manager) + 一些必要的底层库，也就是一个更完整也更大的集成管理工具。
- conda info    //安装成功检查 
- conda env list    //conda虚拟环境list
- conda create —name tensorflow  —clone root    //建立新的conda虚拟环境“tensorflow” 
- source activate tensorflow    //激活“tensorflow”环境

2、TensorFlow安装
- 选择版本-清华镜像：https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/
(tensorflow)$ pip install --ignore-installed —upgrade https://mirrors.tuna.tsinghua.edu.cn/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl
- python
- import tensorflow as tf    //tensorflow安装成功检查
- source deactivate    //关闭“当前tensorflow”环境

# 二、Object Detection for Mac 使用
0、环境准备
https://github.com/tensorflow/models/tree/master/research/object_detection
Tensorflow对象检测API依赖于以下项： 
- Protobuf 2.6 
- Pillow 1.0 
- lxml 
- tf Slim (which is included in the “tensorflow/models” checkout) 
- Jupyter notebook 
- Matplotlib 
- Tensorflow 

以下是安装步骤： 
1、TensorFlow > 1.0 安装或更新
我是pip安装的tensorflow1.2版本，1以下版本好像不兼容该API，命令如下： 
sudo pip install –upgrade  
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp27-cp27mu-manylinux1_x86_64.whl

//一定要先更新tensorflow-1.3，1.0版本会报错InvalidArgumentError
//InvalidArgumentError: NodeDef mentions attr 'data_format'
sudo pip install __upgrade tensorflow


2、Object Detection model下载
Tensorflow Object Detection API的model是在guthub上下载 https://github.com/tensorflow/models ，并解压

//切换目录到tensorflow／models／research
cd /Users/wangyezzz/GitHub/models/research

//激活“tensorflow”环境
source activate tensorflow 


3、其余的库可以通过apt-get安装： 
sudo apt-get install protobuf-compiler python-pil python-lxml 
sudo pip install jupyter 
sudo pip install matplotlib 

以上命令也可以使用以下四条pip命令代替： 
sudo pip install pillow 
sudo pip install lxml 
sudo pip install jupyter 
sudo pip install matplotlib 

注：安装jupyter时可能遇到错误，更新一下pip再安装，sudo -H pip install –upgrade pip 

4、Protobufs安装&编译object_detection/protos
Tensorflow Object Detection API使用Protobufs来配置模型和训练参数。在使用框架之前，必须编译Protobuf库。这应该通过从下载解压的models/目录运行以下命令来完成： 
protoc object_detection/protos/*.proto –python_out=.

Mac安装Protobuf
1）安装 Homebrew／brew：https://brew.sh/index_zh-cn.html
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)” 
使用 Homebrew 安装 Apple 没有预装但 你需要的东西。

2）使用brew安装protoc
brew install protobuf

3）检测protoc 指令是否可以使用    终端输入 protoc —version

Protobuf编译*.proto文件
将.proto文件编译成.h和.m文件

终端输入指令: 
bogon:~ muyang$ protoc ./msg.proto --objc_out=./

详解： protoc  指protoc指令
　　　 /msg.proto  指源码文件所在的路径
　　　--objc_out= 指输出OC文件
　   　./指编译完成后的.h和.m存放的路径　

5、Python模块配置PythonPath环境变量
//当在本地运行时，models /和slim目录应该附加到PYTHONPATH。这可以通过从models /运行以下来完成： 
//注意：此命令需要从您启动的每个新终端运行。如果您想避免手动运行，可以将其作为新行添加到～/ .bashrc文件的末尾
export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim 

//查看pythonpath
echo $PYTHONPATH 

//自己用户的~/.bash_profile中配置,以后python就会自动找到你编写的python模块了
vi ~/.bash_profile

python_tensorflow_models
export PYTHONPATH=$PYTHONPATH:/Users/wangyezzz/GitHub/models/research:/Users/wangyezzz/GitHub/models/research/slim

//生效
source ~/.bash_profile


6、安装完毕：测试一下
至此安装完毕，可以通过运行以下命令来测试是否正确安装了Tensorflow Object Detection API： 
python object_detection / builders / model_builder_test.py

输出OK表示设置完成


7、运行detection demo *.ipynb
jupyter-notebook  （--allow-root）
object_detection_tutorial.ipynb
python /Users/wangyezzz/Desktop/object_detection:g3doc:running_pets/object_detection_tutorial_wy.py


这儿我测试了SSDmobilenet和SSDinception等模型，提前下载好模型，修改里面的代码，将图片测试改为视频读取测试，具体如下： 

下载模型（先下载一个ssd_mobilenet_v1_coco，也是速度最快的）： 
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md 

下载的ssd_mobilenet_v1_coco_11_06_2017.tar.gz放在models/object_detection目录
修改代码，测试SSDmobilenet代码位于 

https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb 


访问：http://127.0.0.1:8888/ 打开object_detection_tutorial.ipynb。 
http://127.0.0.1:8888/notebooks/object_detection_tutorial.ipynb 



默认是处理 object_detection/test_images 文件夹下的image1.jpg、image2.jpg，如果想识别其他图像可以把倒数第二个Cell的代码修改一下： 
Python代码  
1. TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  
2. TEST_IMAGE_PATHS = ['<your image path>']  


在最后一个cell里添加2行代码： 
Python代码  
1. plt.figure(figsize=IMAGE_SIZE)  
2. plt.imshow(image_np)  

-> 
Python代码  
1. print(image_path.split('.')[0]+'_labeled.jpg') Add  
2. plt.figure(figsize=IMAGE_SIZE, dpi=300) # Modify  
3. plt.imshow(image_np)  
4. plt.savefig(image_path.split('.')[0] + '_labeled.jpg') # Add  



然后从头到尾挨个执行每个Cell后等结果。（Download Model那部分代码需要从网上下载文件比较慢！） 



执行完成后在object_detection/test_images 文件夹下就能看到结果图了。 

比较一下官方提供的检测结果图，可见和机器于很大关系： 



8、代码中文说明：
Google发布了新的TensorFlow物体检测API，包含了预训练模型，一个发布模型的jupyter notebook，一些可用于使用自己数据集对模型进行重新训练的有用脚本。
使用该API可以快速的构建一些图片中物体检测的应用。这里我们一步一步来看如何使用预训练模型来检测图像中的物体。
首先我们载入一些会使用的库
接下来进行环境设置
物体检测载入
准备模型
变量  任何使用export_inference_graph.py工具输出的模型可以在这里载入，只需简单改变PATH_TO_CKPT指向一个新的.pb文件。这里我们使用“移动网SSD”模型。
下载模型
载入标签图
标签图将索引映射到类名称，当我们的卷积预测5时，我们知道它对应飞机。这里我们使用内置函数，但是任何返回将整数映射到恰当字符标签的字典都适用。
在载入模型部分可以尝试不同的侦测模型以比较速度和准确度，将你想侦测的图片放入TEST_IMAGE_PATHS中运行即可。


9、代码改为py运行代码文件：

从ImageNet中取一张图2008_004037.jpg测试，然后把 object_detection_tutorial.ipynb 里的代码改成可直接运行代码 

Shell代码  
vi object_detect_demo.py  
python object_detect_demo.py  

Python代码（略）  


10、模型训练和评估
今天我们利用 TF Object Detection API 从头训练一个 Faster RCNN 目标检测模型，使用 ResNet-152 网络和 Oxford-IIIT 宠物数据集。

1）数据集 转换 TFRecord格式
将jpg图片数据转换成TFRecord数据。 
Shell代码  
cd /usr/local/tensorflow2/tensorflow-models/object_detection  
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz  
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz  
tar -zxvf annotations.tar.gz  
tar -zxvf images.tar.gz  
python create_pet_tf_record.py --data_dir=`pwd` --output_dir=`pwd`  
images里全是已经标记好的jpg图片。执行完成后，会在当前目录下生成2个文件：pet_train.record、pet_val.record。 


2）配置pipeline 
在object_detection/samples下有各种模型的通道配置，复制一份出来用。 
Shell代码  
// wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz  
// tar -zxvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz  
cp samples/configs/faster_rcnn_resnet101_pets.config mypet.config  
vi mypet.config  

修改PATH_TO_BE_CONFIGURED、训练steps、fine_tune_checkpoint

第一种： 
 //# fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  from_detection_checkpoint: false
第二种：
from_detection_checkpoint设置为true，fine_tune_checkpoint需要设置检查点的路径。采用别人训练出来的checkpoint可以减少训练时间。 
检查点的下载地址参考： 
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md 

3）训练评估 
Shell代码  
mkdir -p /usr/local/tensorflow2/tensorflow-models/object_detection/model/train  
mkdir -p /usr/local/tensorflow2/tensorflow-models/object_detection/model/eval  


-- 训练 -- 
Shell代码  
python object_detection/train.py --logtostderr --pipeline_config_path='/usr/local/tensorflow2/tensorflow-models/object_detection/mypet.config' --train_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/train'  

-- 评估 -- 
Shell代码  
python object_detection/eval.py --logtostderr --pipeline_config_path='/usr/local/tensorflow2/tensorflow-models/object_detection/mypet.config' --checkpoint_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/train' --eval_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/eval'  

eval文件夹下会生成以下文件，一个文件对应一个image： 
events.out.tfevents.1499152949.localhost.localdomain 
events.out.tfevents.1499152964.localhost.localdomain 
events.out.tfevents.1499152980.localhost.localdomain 

-- 查看结果 -- 
Shell代码  
tensorboard --logdir=/usr/local/tensorflow/tensorflow-models/object_detection/model/  

 train和eval执行后直到终止命令前一直运行 
 训练、评估、查看可以开3个终端分别同时运行 


# 三、视频物体识别
谷歌在github上公布了此项目的完整代码，接下来我们将在现有代码基础上添加相应模块实现对于视频中物体的识别。

第一步：下载opencv的cv2包
在Python官网即可下载opencv相关库，点击此处直接进入。 

博主安装的版本如下： 


下载完成后，在cmd中执行安装命令
pip install opencv_python-3.2.0.8-cp35-cp35m-win_amd64.whl
安装完成后，进入IDLE输入命令
import cv2
若未报错，则opencv-python库成功导入，环境搭配成功。


第二步：在原代码中引入cv2包

第三步：添加视频识别代码 

主要步骤如下： 

1.使用 VideoFileClip 函数从视频中抓取图片。 

2.用fl_image函数将原图片替换为修改后的图片，用于传递物体识别的每张抓取图片。 

3.所有修改的剪辑图像被组合成为一个新的视频。

将识别完毕的视频导为gif格式，并保存至object_detection文件夹。
