import os
import sys
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

import glob
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import pickle
import random
from scipy import misc
from PIL import Image, ImageDraw, ImageFont

class datahelper(Dataset):

    def __init__(self, image_path,label_path):
        self.image_path=image_path
        self.label_path=label_path
        if not os.path.exists('data/cache'):
            os.mkdir("data/cache")
        if os.path.exists('data/custom/labels/core_battery00000003.txt') == False:
        # if os.path.exists('data/custom/labels/core_battery0000000008.txt') == False:
        #     if os.path.exists("data/cache/class_url.pkl"): #如果存在样本目录
        #         self.C1 = pickle.load(open("data/cache/class_url.pkl", "rb")) #C1加载样本目录
        #     else:
            if True:
                self.C1 = self.do_url() #C1格式为[图片，x1，x2，x3，x4]
                pickle.dump(self.C1, open("data/cache/class_url.pkl", "wb")) #将c1写入文件

            print(f"class : {len(self.C1)}") #打印样本的数量

            self.segment() #获得相对坐标处理后的label


    def do_url(self):
        """
        :return: 样本图片及其框的绝对位置
        """
        p = self.label_path+os.sep+ "*.txt"
        # print(p)
        c1_path = glob.glob(p) #自动匹配找出所有的正样本label
        # print(c1_path)
        C1 = [] #样本空序列
        for t in c1_path: #遍历c1_path的所有文件
            # print(t)
            with open(t, "r", encoding='utf-8') as f: #逐一打开文件
                py = f.readlines() #按行进行文件读取
                for oo in py:
                    # print("读取信息为:")
                    # print(oo)
                    oo = oo.split(" ") #进行文件标签分割
                    rec = (float(oo[2]), float(oo[3]), float(oo[4]), float(oo[5])) #记录
                    (filepath, tempfilename) = os.path.split(t)
                    tempfilename=tempfilename.replace("txt","jpg")
                    file_dir=self.image_path+os.sep+tempfilename
                    if oo[1] == "带电芯充电宝":
                        class_name=0;
                    else:
                        class_name=1;
                    C1.append([file_dir, rec,class_name]) #将带芯图片的矩阵加入到C1中
                    # print('添加的信息为')
                    # print([file_dir, rec,class_name])
        return C1

    def segment(self):
        def text_save(filename, data):#filename为写入txst文件的路径，data为要写入数据列表.
            if not os.path.exists('data/custom/labels'):
                os.mkdir("data/custom/labels")
            file = open(filename,'w+')
            for i in range(len(data)):
                s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
                # print(s)
                s = s.replace("'",'').replace(',','') +' '   #去除单引号，逗号，每行末尾追加换行符
                file.writelines(s)
            file.writelines("\n")
            file.close()
        step = 0 #步长
        print("………………请稍等………………")
        for t in self.C1: #遍历正样本，C1为[图片名称，xmin,ymin,xmax,ymax]
            # print("读取的信息为")
            # print(t)
            path = t[0] #图片位置
            # print("图片位置为")
            # print(path)
            rec = t[1] #方框位置
            class_name=t[2]

            lena = cv2.imread(path)  # 读取和代码处于同一目录下的图片
            h=lena.shape[0]
            w=lena.shape[1]
            # print('高和宽为')
            # print(h,w)
            x1 = int(rec[0]) / w #确定x最小值
            if x1 > 1:
                continue
            if x1 < 0:
                x1 = 0
            y1 = int(rec[1]) / h #确定y最小值
            if y1>1:
                continue
            if y1 < 0:
                y1 = 0
            x2 = int(rec[2]) / w
            if x2 > 1:
                x2 = 1
            y2 = int(rec[3]) / h
            if y2 > 1:
                y2 = 1
            # print("相对的位置为")
            # print(x1,y1,x2,y2)
            del lena #删除原图
            (filepath, tempfilename) = os.path.split(path)
            tempfilename=tempfilename.replace("jpg","txt")
            # file_dir='data/custom/labels'+os.sep+tempfilename
            file_dir='data/custom/labels'+os.sep+tempfilename

            a=[class_name,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
            # print("保存的信息为")
            # print(a)
            step += 1
            text_save(file_dir,a)

