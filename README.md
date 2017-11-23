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

# Object Detection for Mac 使用
一、环境准备
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

#python_tensorflow_models
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
1. # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  
2. TEST_IMAGE_PATHS = ['<your image path>']  


在最后一个cell里添加2行代码： 
Python代码  
1. plt.figure(figsize=IMAGE_SIZE)  
2. plt.imshow(image_np)  

-> 
Python代码  
1. print(image_path.split('.')[0]+'_labeled.jpg') # Add  
2. plt.figure(figsize=IMAGE_SIZE, dpi=300) # Modify  
3. plt.imshow(image_np)  
4. plt.savefig(image_path.split('.')[0] + '_labeled.jpg') # Add  



然后从头到尾挨个执行每个Cell后等结果。（Download Model那部分代码需要从网上下载文件比较慢！） 



执行完成后在object_detection/test_images 文件夹下就能看到结果图了。 
image1_labeled.jpg 
image2_labeled.jpg 



比较一下官方提供的检测结果图，可见和机器于很大关系： 



8、代码中文说明：
Google发布了新的TensorFlow物体检测API，包含了预训练模型，一个发布模型的jupyter notebook，一些可用于使用自己数据集对模型进行重新训练的有用脚本。
使用该API可以快速的构建一些图片中物体检测的应用。这里我们一步一步来看如何使用预训练模型来检测图像中的物体。
首先我们载入一些会使用的库
[python] view plain copy
1. import numpy as np  
2. import os  
3. import six.moves.urllib as urllib  
4. import sys  
5. import tarfile  
6. import tensorflow as tf  
7. import zipfile  
8.   
9. from collections import defaultdict  
10. from io import StringIO  
11. from matplotlib import pyplot as plt  
12. from PIL import Image  


接下来进行环境设置
[python] view plain copy
1. %matplotlib inline  
2. sys.path.append("..")  
物体检测载入
[python] view plain copy
1. from utils import label_map_util  
2.   
3. from utils import visualization_utils as vis_util  
准备模型
变量  任何使用export_inference_graph.py工具输出的模型可以在这里载入，只需简单改变PATH_TO_CKPT指向一个新的.pb文件。这里我们使用“移动网SSD”模型。
[python] view plain copy
1. MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'  
2. MODEL_FILE = MODEL_NAME + '.tar.gz'  
3. DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  
4.   
5. PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'  
6.   
7. PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')  
8.   
9. NUM_CLASSES = 90  
下载模型
[python] view plain copy
1. opener = urllib.request.URLopener()  
2. opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)  
3. tar_file = tarfile.open(MODEL_FILE)  
4. for file in tar_file.getmembers():  
5.     file_name = os.path.basename(file.name)  
6.     if 'frozen_inference_graph.pb' in file_name:  
7.         tar_file.extract(file, os.getcwd())  
将（frozen）TensorFlow模型载入内存
[python] view plain copy
1. detection_graph = tf.Graph()  
2. with detection_graph.as_default():  
3.     od_graph_def = tf.GraphDef()  
4.     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  
5.         serialized_graph = fid.read()  
6.         od_graph_def.ParseFromString(serialized_graph)  
7.         tf.import_graph_def(od_graph_def, name='')  


载入标签图
标签图将索引映射到类名称，当我们的卷积预测5时，我们知道它对应飞机。这里我们使用内置函数，但是任何返回将整数映射到恰当字符标签的字典都适用。
[python] view plain copy
1. label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  
2. categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)  
3. category_index = label_map_util.create_category_index(categories)  
辅助代码
[python] view plain copy
1. def load_image_into_numpy_array(image):  
2.   (im_width, im_height) = image.size  
3.   return np.array(image.getdata()).reshape(  
4.       (im_height, im_width, 3)).astype(np.uint8)  
检测
[python] view plain copy
1. PATH_TO_TEST_IMAGES_DIR = 'test_images'  
2. TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  
3. IMAGE_SIZE = (12, 8)  
[python] view plain copy
1. with detection_graph.as_default():  
2.   
3.   with tf.Session(graph=detection_graph) as sess:  
4.     for image_path in TEST_IMAGE_PATHS:  
5.       image = Image.open(image_path)  
6.       # 这个array在之后会被用来准备为图片加上框和标签  
7.       image_np = load_image_into_numpy_array(image)  
8.       # 扩展维度，应为模型期待: [1, None, None, 3]  
9.       image_np_expanded = np.expand_dims(image_np, axis=0)  
10.       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  
11.       # 每个框代表一个物体被侦测到.  
12.       boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  
13.       # 每个分值代表侦测到物体的可信度.  
14.       scores = detection_graph.get_tensor_by_name('detection_scores:0')  
15.       classes = detection_graph.get_tensor_by_name('detection_classes:0')  
16.       num_detections = detection_graph.get_tensor_by_name('num_detections:0')  
17.       # 执行侦测任务.  
18.       (boxes, scores, classes, num_detections) = sess.run(  
19.           [boxes, scores, classes, num_detections],  
20.           feed_dict={image_tensor: image_np_expanded})  
21.       # 图形化.  
22.       vis_util.visualize_boxes_and_labels_on_image_array(  
23.           image_np,  
24.           np.squeeze(boxes),  
25.           np.squeeze(classes).astype(np.int32),  
26.           np.squeeze(scores),  
27.           category_index,  
28.           use_normalized_coordinates=True,  
29.           line_thickness=8)  
30.       plt.figure(figsize=IMAGE_SIZE)  
31.       plt.imshow(image_np)  
在载入模型部分可以尝试不同的侦测模型以比较速度和准确度，将你想侦测的图片放入TEST_IMAGE_PATHS中运行即可。


9、代码改为py运行代码文件：

从ImageNet中取一张图2008_004037.jpg测试，然后把 object_detection_tutorial.ipynb 里的代码改成可直接运行代码 

Shell代码  
1. # vi object_detect_demo.py  
2. # python object_detect_demo.py  


Python代码  
1. import numpy as np  
2. import os  
3. import six.moves.urllib as urllib  
4. import sys  
5. import tarfile  
6. import tensorflow as tf  
7. import zipfile  
8. import matplotlib  
9.   
10. # Matplotlib chooses Xwindows backend by default.  
11. matplotlib.use('Agg')  
12.   
13. from collections import defaultdict  
14. from io import StringIO  
15. from matplotlib import pyplot as plt  
16. from PIL import Image  
17. from utils import label_map_util  
18. from utils import visualization_utils as vis_util  
19.   
20. ##################### Download Model  
21. # What model to download.  
22. MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'  
23. MODEL_FILE = MODEL_NAME + '.tar.gz'  
24. DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  
25.   
26. # Path to frozen detection graph. This is the actual model that is used for the object detection.  
27. PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'  
28.   
29. # List of the strings that is used to add correct label for each box.  
30. PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')  
31.   
32. NUM_CLASSES = 90  
33.   
34. # Download model if not already downloaded  
35. if not os.path.exists(PATH_TO_CKPT):  
36.     print('Downloading model... (This may take over 5 minutes)')  
37.     opener = urllib.request.URLopener()  
38.     opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)  
39.     print('Extracting...')  
40.     tar_file = tarfile.open(MODEL_FILE)  
41.     for file in tar_file.getmembers():  
42.         file_name = os.path.basename(file.name)  
43.         if 'frozen_inference_graph.pb' in file_name:  
44.             tar_file.extract(file, os.getcwd())  
45. else:  
46.     print('Model already downloaded.')  
47.   
48. ##################### Load a (frozen) Tensorflow model into memory.  
49. print('Loading model...')  
50. detection_graph = tf.Graph()  
51.   
52. with detection_graph.as_default():  
53.     od_graph_def = tf.GraphDef()  
54.     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  
55.         serialized_graph = fid.read()  
56.         od_graph_def.ParseFromString(serialized_graph)  
57.         tf.import_graph_def(od_graph_def, name='')  
58.   
59. ##################### Loading label map  
60. print('Loading label map...')  
61. label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  
62. categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)  
63. category_index = label_map_util.create_category_index(categories)  
64.   
65. ##################### Helper code  
66. def load_image_into_numpy_array(image):  
67.   (im_width, im_height) = image.size  
68.   return np.array(image.getdata()).reshape(  
69.       (im_height, im_width, 3)).astype(np.uint8)  
70.   
71. ##################### Detection  
72. # Path to test image  
73. TEST_IMAGE_PATH = 'test_images/2008_004037.jpg'  
74.   
75. # Size, in inches, of the output images.  
76. IMAGE_SIZE = (12, 8)  
77.   
78. print('Detecting...')  
79. with detection_graph.as_default():  
80.   with tf.Session(graph=detection_graph) as sess:  
81.     print(TEST_IMAGE_PATH)  
82.     image = Image.open(TEST_IMAGE_PATH)  
83.     image_np = load_image_into_numpy_array(image)  
84.     image_np_expanded = np.expand_dims(image_np, axis=0)  
85.     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  
86.     boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  
87.     scores = detection_graph.get_tensor_by_name('detection_scores:0')  
88.     classes = detection_graph.get_tensor_by_name('detection_classes:0')  
89.     num_detections = detection_graph.get_tensor_by_name('num_detections:0')  
90.     # Actual detection.  
91.     (boxes, scores, classes, num_detections) = sess.run(  
92.         [boxes, scores, classes, num_detections],  
93.         feed_dict={image_tensor: image_np_expanded})  
94.     # Print the results of a detection.  
95.     print(scores)  
96.     print(classes)  
97.     print(category_index)  
98.     # Visualization of the results of a detection.  
99.     vis_util.visualize_boxes_and_labels_on_image_array(  
100.         image_np,  
101.         np.squeeze(boxes),  
102.         np.squeeze(classes).astype(np.int32),  
103.         np.squeeze(scores),  
104.         category_index,  
105.         use_normalized_coordinates=True,  
106.         line_thickness=8)  
107.     print(TEST_IMAGE_PATH.split('.')[0]+'_labeled.jpg')  
108.     plt.figure(figsize=IMAGE_SIZE, dpi=300)  
109.     plt.imshow(image_np)  
110.     plt.savefig(TEST_IMAGE_PATH.split('.')[0] + '_labeled.jpg')  



10、模型训练和评估
今天我们利用 TF Object Detection API 从头训练一个 Faster RCNN 目标检测模型，使用 ResNet-152 网络和 Oxford-IIIT 宠物数据集。

1）数据集 转换 TFRecord格式
将jpg图片数据转换成TFRecord数据。 
Shell代码  
1. # cd /usr/local/tensorflow2/tensorflow-models/object_detection  
2. # wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz  
3. # wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz  
4. # tar -zxvf annotations.tar.gz  
5. # tar -zxvf images.tar.gz  
6. # python create_pet_tf_record.py --data_dir=`pwd` --output_dir=`pwd`  
images里全是已经标记好的jpg图片。执行完成后，会在当前目录下生成2个文件：pet_train.record、pet_val.record。 


2）配置pipeline 
在object_detection/samples下有各种模型的通道配置，复制一份出来用。 
Shell代码  
1. // wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz  
2. // tar -zxvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz  
3. # cp samples/configs/faster_rcnn_resnet101_pets.config mypet.config  
4. # vi mypet.config  

修改PATH_TO_BE_CONFIGURED、训练steps、fine_tune_checkpoint

第一种： 
 # fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  from_detection_checkpoint: false
第二种：
from_detection_checkpoint设置为true，fine_tune_checkpoint需要设置检查点的路径。采用别人训练出来的checkpoint可以减少训练时间。 
检查点的下载地址参考： 
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md 

3）训练评估 
Shell代码  
1. # mkdir -p /usr/local/tensorflow2/tensorflow-models/object_detection/model/train  
2. # mkdir -p /usr/local/tensorflow2/tensorflow-models/object_detection/model/eval  


-- 训练 -- 
Shell代码  
1. # python object_detection/train.py \  
2.      --logtostderr \  
3.      --pipeline_config_path='/usr/local/tensorflow2/tensorflow-models/object_detection/mypet.config' \  
4.      --train_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/train'  

-- 评估 -- 
Shell代码  
1. # python object_detection/eval.py \  
2.     --logtostderr \  
3.     --pipeline_config_path='/usr/local/tensorflow2/tensorflow-models/object_detection/mypet.config' \  
4.     --checkpoint_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/train' \  
5.     --eval_dir='/usr/local/tensorflow2/tensorflow-models/object_detection/model/eval'  

eval文件夹下会生成以下文件，一个文件对应一个image： 
events.out.tfevents.1499152949.localhost.localdomain 
events.out.tfevents.1499152964.localhost.localdomain 
events.out.tfevents.1499152980.localhost.localdomain 

-- 查看结果 -- 
Shell代码  
1. # tensorboard --logdir=/usr/local/tensorflow/tensorflow-models/object_detection/model/  

*** train和eval执行后直到终止命令前一直运行 
*** 训练、评估、查看可以开3个终端分别同时运行 


# 视频物体识别
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
在原版代码基础上，在最后面依次添加如下代码（可从完整代码 处复制，但需要作出一些改变，当然也可以直接从下文复制修改后的代码）： 


# Import everything needed to edit/save/watch video clips
import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from IPython.display import HTML
此处会下载一个剪辑必备的程序ffmpeg.win32.exe，内网下载过程中容易断线，可以使用下载工具下载完然后放入如下路径： 
C:\Users\ 用户名 \AppData\Local\imageio\ffmpeg\ffmpeg.win32.exe
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph)
            return image_process

white_output = 'video1_out.mp4'
clip1 = VideoFileClip("video1.mp4").subclip(25,30)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
%time white_clip.write_videofile(white_output, audio=False)
其中video1.mp4已经从电脑中上传至object_detection文件夹，subclip（25,30）代表识别视频中25-30s这一时间段。
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
原版视频： 

展示识别完毕的视频： 


from moviepy.editor import *
clip1 = VideoFileClip("video1_out.mp4")
clip1.write_gif("final.gif")

将识别完毕的视频导为gif格式，并保存至object_detection文件夹。
至此，快速教程结束。各位应该都能使用谷歌开放的API实现了视频物体识别。



整体代码：

# coding: utf-8
# In[4]:

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# In[6]:


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# In[7]:


NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[8]:


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


# In[9]:


# First test on images
# PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[11]:


from PIL import Image
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    plt.imshow(image_np)
    print(image.size, image_np.shape)


# In[12]:


#Load a frozen TF model 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[13]:


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_process = detect_objects(image_np, sess, detection_graph)
            print(image_process.shape)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_process)
      


# In[26]:


# Import everything needed to edit/save/watch video clips
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[27]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph)
            return image_process


# In[48]:


white_output = 'video1_out.mp4'
clip1 = VideoFileClip("video1.mp4").subclip(0,4)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[49]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[50]:


# Merge videos
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# clip1 = VideoFileClip("video1_out.mp4")
# clip2 = VideoFileClip("fruits1_out.mp4")
# clip3 = VideoFileClip("dog_out.mp4")
# final_clip = concatenate_videoclips([clip1,clip2,clip3], method="compose")
# final_clip.write_videofile("my_concatenation.mp4",bitrate="5000k")


# In[51]:


from moviepy.editor import *
# clip = VideoFileClip("my_concatenation.mp4")
clip = VideoFileClip("video1_out.mp4")
clip.write_gif("final.gif")



摄像头识别：
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time  

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
#MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/hmw/tensorflow/models/object_detection/data', 'mscoco_label_map.pbtxt')

#extract the ssd_mobilenet
start = time.clock()
NUM_CLASSES = 90
opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
end= time.clock()
print 'load the model',(end-start)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture(0)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      writer = tf.summary.FileWriter("logs/", sess.graph)  
      sess.run(tf.global_variables_initializer())  
      while(1):
    start = time.clock()
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        image_np=frame
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=6)
    end = time.clock()
    print 'frame:',1.0/(end - start)
    #print 'frame:',time.time() - start
    cv2.imshow("capture", image_np)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

修改后在models/object_detection目录下运行(否则会报错)，先添加root权限（否则会报错），此测试需要提前安装好opencv2，（若没有安装运行命令：sudo apt-get install python-opencv 即可安装）接上摄像头然后python运行上述代码webcamtest.py，得到结果。测试时如果出现错误HIGHGUI ERROR: libv4l unable to ioctl VIDIOCSPICT…，将命令窗口关闭重新开启即可解决
