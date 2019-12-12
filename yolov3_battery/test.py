from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
from datahelper_4 import datahelper

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
#evaluate(model,path=valid_path, iou_thres=opt.iou_thres,conf_thres=opt.conf_thres,nms_thres=opt.nms_thres,img_size=opt.img_size,batch_size=8,)
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
def make_test_path(dir,txt_path):
    """

    :param dir: 测试文件根目录
    :param txt_path: 测试文件名称
    :return: 带位置的测试文件目录
    """
    with open(txt_path, "r", encoding='utf-8') as f: #逐一打开文件
        if not os.path.exists('data/custom'):
            os.mkdir("data/custom")
        save_file="data/custom/valid.txt"
        file = open(save_file,'w+')
        py = f.readlines() #按行进行文件读取

        for oo in py:
            file_dir=dir+os.sep+oo.strip()+".jpg"
            file.writelines(file_dir)
            file.write("\n")
        file.close()
        f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
     parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_30.pth", help="path to weights file")
#    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.005, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    image_path = data_config["image_path"]
    test_path=data_config["test"]
    label_path=data_config["label_path"]
    class_names = load_classes(data_config["names"])
    make_test_path(image_path,test_path)
    valid_path="data/custom/valid.txt"

    datahelper(image_path,label_path)
    print("ok")
    # Initiate model
    model = Darknet(opt.model_def).to(device) #建立darknet模型
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path) #载入darknet参数
    else:
        # Load checkpoint weights
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.weights_path)) #载入普通参数
        else:
            model.load_state_dict(torch.load(opt.weights_path,map_location=lambda storage, loc: storage))
       

    print("Compute mAP...") #计算mAP

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    ) #计算精确率度，召回率，AP，f1，和ap_class

    print("Average Precisions:") #计算每个类的平均精确率
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}") #计算平均AP
