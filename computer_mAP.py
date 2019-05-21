from voc_eval import voc_eval

import os

current_path = os.getcwd()
results_path = current_path + "/results"
sub_files = os.listdir(results_path)

mAP = []
for i in range(len(sub_files)):
    class_name = sub_files[i].split(".txt")[0]
    # class_name = sub_files[0].split("_")[-1].split(".txt")[0]
    rec, prec, ap = voc_eval('results/{}.txt',
                             'VOCdevkit/VOC2012/Annotations/{}.xml',
                             'VOCdevkit/VOC2012/ImageSets/Main/test.txt',
                             class_name, '.')
    print("{} :\t {} ".format(class_name, ap))
    mAP.append(ap)

mAP = tuple(mAP)

print("***************************")
print("mAP :\t {}".format(float(sum(mAP) / len(mAP))))
