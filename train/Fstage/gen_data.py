#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 10:42 AM
# @Author  : Ject.Y
# @Site    :
# @File    : gen_data.py
# @Software: PyCharm
# @Copyright: BSD 2.0
# @Function : To generating training data
# How to run : run

import numpy as np
import os
import cv2
import sys
import random
import progressbar as pb

sys.path.append('../train')
from common import functions as fs



#========================== To be configure ====================
TXT = 'WIDER_train/wider_face_train.txt'
BASE_PATH = 'data_images'
MEG_PATH = BASE_PATH + '/neg'
POS_PATH = BASE_PATH + '/pos'
PART_PATH = BASE_PATH + '/part'
#===============================================================
with open(TXT, 'r') as T:
    tlines = T.readlines()

if __name__ == '__main__':
    NID = 0
    PID = 0
    PAID = 0
    fs.mkdirs([BASE_PATH, MEG_PATH, POS_PATH, PART_PATH])

    FN = open(os.path.join(BASE_PATH, 'neg.txt'), 'w')
    FP = open(os.path.join(BASE_PATH, 'pos.txt'), 'w')
    FPA = open(os.path.join(BASE_PATH, 'part.txt'), 'w')

    widgets = ['Time for data: ', pb.Percentage(), ' ',
               pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
    # initialize timer
    timer = pb.ProgressBar(widgets=widgets, maxval=len(tlines)).start()
    nums = len(tlines)
    for ids, l in enumerate(tlines):
        timer.update(ids)
        widgets[0] = 'for %dth image of %d' % (ids + 1, nums)
        l = l.strip().split(' ')
        if len(l) < 3:
            continue
        box = map(float, l[1:])
        boxes = np.array(box, dtype=np.float).reshape(-1, 4)
        img = cv2.imread(os.path.join('WIDER_train/images',l[0]+'.jpg'))
        assert img.shape[0] > 0 and img.shape[1] > 0, 'IMAGE %s READ ERROR!!' % l[0]

        sposid = 0
        sneid = 0
        spaid = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            loop = 0
            while loop < 300:
                if sposid > 50 and sneid > 150 and spaid > 50:
                    break
                loop += 1
                scale = random.uniform(0.5, 1.5)
                xoffset = random.uniform(-0.2, 0.5)
                yoffset = random.uniform(-0.2, 0.5)
                size_w = w * scale
                size_h = size_w
                x1_ = x1 + w * xoffset
                y1_ = y1 + h * yoffset
                x2_ = x1_ + size_w
                y2_ = y1_ + size_h
                if y2_ - y1_ < 24 or x2_ - x1_ < 24 or x1_ < 0 or y1_ < 0 or x2_ > img.shape[1] or y2_ > img.shape[0]:
                    continue
                box_ = np.array([x1_, y1_, x2_, y2_], np.float32)
                cropped_im = img[int(round(y1_)):int(round(y2_)), int(round(x1_)):int(round(x2_)), :]
                iou_score = fs.IoUs(box_, boxes)
                if np.max(iou_score) < 0.3:
                    if sneid > 150:
                        continue
                    # cropped_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_file = os.path.join(MEG_PATH, "%s.jpg" % NID)
                    FN.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, cropped_im)
                    NID += 1
                    sneid += 1
                else:
                    offset_x1 = (x1 - x1_) / float(size_w)
                    offset_y1 = (y1 - y1_) / float(size_h)
                    offset_x2 = (x2 - x2_) / float(size_w)
                    offset_y2 = (y2 - y2_) / float(size_h)
                    if fs.IoU(box_, box) >= 0.65:
                        if sposid > 50:
                            continue
                        save_file = os.path.join(POS_PATH, "%s.jpg" % PID)
                        FP.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, cropped_im)
                        PID += 1
                        sposid += 1
                    elif fs.IoU(box_, box) >= 0.4:
                        if spaid > 50:
                            continue
                        save_file = os.path.join(PART_PATH, "%s.jpg" % PAID)
                        FPA.write(
                            save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, cropped_im)
                        PAID += 1
                        spaid += 1

    timer.finish()

    quit()
