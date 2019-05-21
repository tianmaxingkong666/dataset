import os
import sys, os
pwd = os.getcwd()
import numpy as np
from PIL import Image
from config import yoloCfg,yoloWeights,yoloData,darknetRoot
os.chdir(darknetRoot)
sys.path.append('python')
import darknet as dn
from PIL import Image, ImageFont, ImageDraw
import xml.etree.ElementTree as ET

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect_np(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = array_to_image(image)
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_detections(dets, num)
    return res

import cv2
def to_box(r):
    boxes = []
    scores = []
    classes = []
    for rc in r:
        classes.append(rc[0].decode())
        cx,cy,w,h = rc[-1]
        scores.append(rc[1])
        prob  = rc[1]
        xmin,ymin,xmax,ymax = cx-w/2,cy-h/2,cx+w/2,cy+h/2
        boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
    return classes, boxes, scores
 

import pdb
#dn.set_gpu(0)
net = dn.load_net(yoloCfg.encode('utf-8'), yoloWeights.encode('utf-8'), 0)
meta = dn.load_meta(yoloData.encode('utf-8'))

os.chdir(pwd)

def detect(img):
    # img = Image.open(image_file).convert("RGB")
    img = np.array(img)
    r = detect_np(net, meta, img,thresh=0.1, hier_thresh=0.5, nms=0.8)
    classes, boxes, scores = to_box(r)
    
    return classes, boxes, scores

if __name__ == '__main__':
    #image_file="/data01/project/darknet/VOCdevkit/VOC2012/JPEGImages/989f220b756d60.jpg"
    #classes, boxes, scores = detect(image_file)
    #print("classes:%s"%(classes))
    #print("boxes:%s"%(boxes))
    #print("scores:%s"%(scores))

    test_files = open("darknet/test.txt", "r")
    content = test_files.readlines()
    length = len(content)
    acc = 0
    total = 0
    for i in range(length):
        test_file = content[i].replace("\n", "")
        print("processing %s ..." %(test_file))

        xml_file = test_file.replace("JPEGImages", "Annotations").replace("jpg", "xml")
        # 从xml读取车型，坐标数据
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            model_name = obj.find('name').text
        print("the label is %s "%(model_name))

        classes, boxes, scores = detect(test_file)
        if len(classes) > 0:
            pre_cls = classes[0]
            print("the predict is %s"%(pre_cls))
        else:
            pre_cls = "null"
            print("the predict is null")

        if pre_cls == model_name:
            acc += 1

        total += 1
        print("==================================================")

    acc = acc / total
    print("the test acc is %s" %(acc))
 

