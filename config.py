import os
darknetRoot = os.path.join(os.path.curdir,"darknet")## yolo 安装目录
pwd = os.getcwd()
yoloCfg = os.path.join(pwd,"darknet","cfg","yolov3-voc2.cfg")
yoloWeights = os.path.join(pwd,"darknet","backup","yolov3-voc.backup")
yoloData = os.path.join(pwd,"darknet","cfg","voc.data")


# darknet score 的阈值
DETECT_THRESHOLD = 0

