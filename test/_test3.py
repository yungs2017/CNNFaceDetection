#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:39 AM
# @Author  : Ject.Y
# @Site    : 
# @File    : _test3.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : Test the third stage
# How to run : run

import os,sys
sys.path.append('/Users/Mac/install_libs/caffe-master/python/')
import caffe
import cv2
from _test1 import NMS,call_camera
from _test2 import detect_rnet

deploy = 'net.prototxt'
caffemodel = '../train/Tstage/models/solver_iter_150000.caffemodel'

net_caffe = caffe.Net(deploy, caffemodel, caffe.TEST)

def detectFace(img):
    """
        detect face by CNN
        :param img: the resized image mat
        :return: the output score of the mat in the CNN
        """
    net_caffe.blobs['X'].reshape(1, 3, 24, 24)
    img = cv2.resize(img, (24, 24))
    img = (img - 127.5) / 127.5
    img = img.transpose((2, 0, 1))
    net_caffe.blobs['X'].data[...] = img
    out = net_caffe.forward()
    cls_prob = out['prob'][0][1]
    return cls_prob, out['conv4-2'][0]


def detect_onet(img):
    rects = detect_rnet(img.copy())
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
            print prop
            x1_offset, y1_offset, x2_offset, y2_offset = offset
            re_ = map(float, re_)
            x1, y1, x2, y2, _ = re_[:5]
            w = x2 - x1
            h = y2 - y1
            x1 += x1_offset * w
            y1 += y1_offset * h
            x2 += x2_offset * w
            y2 += y2_offset * h
            re_2 = [x1, y1, x2, y2, prop]
            rects_new.append(re_2)

    return NMS(rects_new,0.09)

if __name__ == "__main__":
    try:
        call_camera(detect_onet,video='/Users/Mac/Desktop/VID_20180718_195703.mp4'
                ,rote90=True
                )
    except:
        pass
    call_camera(detect_onet,video=0)