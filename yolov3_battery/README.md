# yolov3_battery
本项目为机器学习大作业充电宝样本不均衡问题提交项目，训练模型为yolov3,
参考原型为https://github.com/eriklindernoren/PyTorch-YOLOv3
本项目提供训练完成的模型及测试文件，不提供训练数据集与训练文件，如有需要，请联系本项目作者

## 准备工作
    $ cd yolov3_battery/
    $ sudo pip3 install -r requirements.txt
    
## 下载模型
    (这是一个模型地址）

## 测试准备

打开"/config/custom.data,
将test设置为测试集使用的图片名，内容格式举例："coreless00000001"
将image_path设置为图片所在路径
将label_path设置为图片所需要的标注文件路径



## 开始测试
使用命令行代码
    $ python3 train.py --weights_path 下载好的模型路径




