#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:32 AM
# @Author  : Ject.Y
# @Site    : 
# @File    : _test1.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : Test the First stage of cnn
# How to run : run

import os,sys
sys.path.append('/Users/Mac/install_libs/caffe-master/python/')
import caffe
import cv2
import numpy as np

deploy = 'net.prototxt'
caffemodel = '../train/Fstage/models/solver_iter_150000.caffemodel'


# GPU_ID = 1  # Switch between 0 and 1 depending on the GPU you want to use.
# caffe.set_mode_gpu()
# caffe.set_device(GPU_ID)
net_caffe = caffe.Net(deploy, caffemodel, caffe.TEST)

def scaleImage(img, min):
    """
    Pyramid scale an image
    :param img: the image need to make multi scales
    :return: a list scales of the image
    """
    scales = []
    img_ = img.copy()
    h, w, c = img.shape
    factor = 0.8
    while h >= min and w >= min:
        img_ = cv2.resize(img_, (w, h))
        scales.append(img_)
        h = int(h * factor)
        w = int(w * factor)

    return scales


def detectFaceInput(img):
    strip = 6
    minsize = 24
    scales = scaleImage(img, minsize)
    rects = []
    for i, s in enumerate(scales):
        f = 0.8 ** i
        h_, w_, c_ = s.shape
        net_caffe.blobs['X'].reshape(1, 3, h_, w_)
        img_ = (s - 127.5) / 127.5
        img__ = img_.transpose((2, 0, 1))
        net_caffe.blobs['X'].data[...] = img__
        # caffe.set_device(GPU_ID)
        out_ = net_caffe.forward()
        for (y, x), pscore in np.ndenumerate(out_['prob'][0][1]):
            if pscore > 0.5:
                x1 = x * strip
                y1 = y * strip
                x2 = x1 + minsize
                y2 = y1 + minsize
                x1 = max(0, x1 * (1.0 / f) )
                y1 = max(0, y1 * (1.0 / f) )
                x2 = min(img.shape[1], x2 * (1.0 / f))
                y2 = min(img.shape[0], y2 * (1.0 / f) )

                if x2 - x1 > 24 and y2 - y1 > 24:
                    rects.append([x1, y1, x2, y2, out_['prob'][0][1][y][x],
                                  (x2 - x1) * out_['conv4-2'][0][0][y][x],
                                  (y2 - y1) * out_['conv4-2'][0][1][y][x],
                                  (x2 - x1) * out_['conv4-2'][0][2][y][x],
                                  (y2 - y1) * out_['conv4-2'][0][3][y][x]
                                  ]
                                 )

    rects = NMS(rects, 0.7)
    if len(rects) > 0:
        rects = np.array(rects,dtype=float)
        rects[:,:4] += rects[:,5:]

    return rects


def NMS(rects, threshold):
    rects = sorted(rects, key=getKey, reverse=True)
    cout = len(rects)
    cu = 0
    while cu < cout:
        count_ = cout - cu - 1
        cu_ = cu + 1
        while count_ > 0:
            score = IoU(rects[cu], rects[cu_])
            if score > threshold:
                del rects[cu_]
                cout -= 1
            else:
                cu_ += 1
            count_ -= 1
        cu += 1
    return rects


def getKey(item):
    return item[4]


def IoU(rect_1, rect_2):
    x11 = rect_1[0]  # first rectangle top left x
    y11 = rect_1[1]  # first rectangle top left y
    x12 = rect_1[2]  # first rectangle bottom right x
    y12 = rect_1[3]  # first rectangle bottom right y
    x21 = rect_2[0]  # second rectangle top left x
    y21 = rect_2[1]  # second rectangle top left y
    x22 = rect_2[2]  # second rectangle bottom right x
    y22 = rect_2[3]  # second rectangle bottom right y
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    intersection = x_overlap * y_overlap
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersection
    if union == 0:
        return 0
    else:
        return float(intersection) / union




def call_camera(function,video=0,rote90=False):
    import imutils
    cap = cv2.VideoCapture(video)
    resize_size = 300
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if rote90:
            frame = imutils.rotate(frame, 90)
        h, w, _ = frame.shape
        if w > resize_size:
            f = float(resize_size) / w
            h = h * f
            img = cv2.resize(frame, (resize_size, int(h)))
            rects = function(img)
            f = 1.0 / f
            img = frame
            rects_new = []
            for re in rects:
                re = map(float, re)
                re[0] = re[0] * f
                re[1] = re[1] * f
                re[2] = re[2] * f
                re[3] = re[3] * f
                rects_new.append(re)
                re = map(round,re)
                re = map(int, re)
                cv2.rectangle(img, (int(re[0]), int(re[1])), (int(re[2]), int(re[3])), (0, 255, 0),2)
        cv2.imshow("images", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    call_camera(detectFaceInput)