#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:00 AM
# @Author  : Ject.Y
# @Site    :
# @File    : hard_sample.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : Generate hard-sample data for third stage
# How to run : run

import os
import sys
sys.path.append('caffe/python')
import caffe
import cv2
import numpy as np
import progressbar as pb
sys.path.append('../')
from common import functions as fs
from Sestage.hard_sample import detectFaceInput,deploy,NMS,IoU,IoUs

#========================== To be configure ====================
TXT = '/home/eryuan/WIDER_train/wider_face_train.txt'
BASE_PATH = 'data_images'
MEG_PATH = BASE_PATH + '/neg'
POS_PATH = BASE_PATH + '/pos'
PART_PATH = BASE_PATH + '/part'
caffemodel2 = '../Sestage/models/solver_iter_220000.caffemodel'
#===============================================================

GPU_ID = 0  # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
net_caffe2 = caffe.Net(deploy, caffemodel2, caffe.TEST)


def detectFace(img):
    """
        detect face by CNN
        :param img: the resized image mat
        :return: the output score of the mat in the CNN
        """
    net_caffe2.blobs['X'].reshape(1, 3, 24, 24)
    img = cv2.resize(img, (24, 24))
    img = (img - 127.5) / 127.5
    img = img.transpose((2, 0, 1))
    net_caffe2.blobs['X'].data[...] = img
    out = net_caffe2.forward()
    cls_prob = out['prob'][0][1]
    return cls_prob, out['conv4-2'][0]

def detect_rnet(img):
    rects = detectFaceInput(img.copy())
    if len(rects) <= 0:
        return []
    rects_new = []
    for re in rects:
        re_ = re
        re = map(round,re)
        re = map(int,re)
        re[0] = max(0,re[0])
        re[1] = max(0,re[1])
        re[2] = min(img.shape[1],re[2])
        re[3] = min(img.shape[0],re[3])

        if re[3] - re[1] < 24 or re[2] - re[0] < 24:
            continue
        img_ = img[re[1]:re[3],re[0]:re[2]]
        prop,offset = detectFace(img_)
        if prop > 0.3:
            x1_offset, y1_offset, x2_offset, y2_offset = offset
            re_ = map(float, re_)
            relist = re_[:5]
            x1, y1, x2, y2, _ = relist
            w = x2 - x1
            h = y2 - y1
            x1 += x1_offset * w
            y1 += y1_offset * h
            x2 += x2_offset * w
            y2 += y2_offset * h
            # print [x1[0][0], y1, x2, y2, prop]
            re_2 = [x1[0][0], y1[0][0], x2[0][0], y2[0][0], prop[0][0]]
            rects_new.append(re_2)

    return NMS(rects_new,0.8)


if __name__ == "__main__":
    with open(TXT,'r') as T:
        tlines = T.readlines()

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
        widgets[4] = 'P %d N %d PA %d ' % (PID + 1,NID + 1,PAID + 1)
        l = l.strip().split(' ')
        if len(l) < 3:
            continue
        box = map(float, l[1:])
        boxes = np.array(box, dtype=np.float).reshape(-1, 4)
        img = cv2.imread(os.path.join('WIDER_train/images', l[0] + '.jpg'))
        assert img.shape[0] > 0 and img.shape[1] > 0, 'IMAGE %s READ ERROR!!' % l[0]

        h, w, _ = img.shape
        f = 1.0
        if w > 300:
            f = 300.0 / w
            h = round(h * f)
            img_ = cv2.resize(img, (300, int(h)))

        f = 1.0 / f

        rects = detect_rnet(img_)
        for rebox in rects:
            rebox_ = np.array(rebox,np.float)
            rebox_ = rebox_ * f
            rebox = rebox_.tolist()
            rebox = map(round,rebox)
            rebox = map(int,rebox)
            x1,y1,x2,y2,_ = rebox[:5]
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(img.shape[1],x2)
            y2 = min(img.shape[0],y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            Iou_score = IoUs(rebox, boxes)
            cropped_im = img[y1:y2, x1:x2,:]
            resized_im = cv2.resize(cropped_im, (24, 24))
            # save negative images and write label
            if np.max(Iou_score) < 0.3:
                if NID > (ids + 1) * 12:
                    continue
                # Iou with all gts must below 0.3
                save_file = os.path.join(MEG_PATH, "%s.jpg" % NID)
                FN.write(save_file + ' 0\n')
                cv2.imwrite(save_file, cropped_im)
                NID += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou_score)
                assigned_gt = boxes[idx]
                x1o, y1o, x2o, y2o = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1o - x1) / float(x2 - x1)
                offset_y1 = (y1o - y1) / float(y2 - y1)
                offset_x2 = (x2o - x2) / float(x2 - x1)
                offset_y2 = (y2o - y2) / float(y2 - y1)
                # save positive and part-face images and write labels
                if np.max(Iou_score) >= 0.65:
                    save_file = os.path.join(POS_PATH, "%s.jpg" % PID)
                    FP.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, cropped_im)
                    PID += 1
                elif np.max(Iou_score) >= 0.4:
                    if PAID > (ids + 1) * 6:
                        continue
                    save_file = os.path.join(PART_PATH, "%s.jpg" % PAID)
                    FPA.write(
                        save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, cropped_im)
                    PAID += 1
    timer.finish()
    quit()
