#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:00 AM
# @Author  : Ject.Y
# @Site    :
# @File    : hard_sample.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : Generate hard-sample data for second stage
# How to run : run

import os
import sys
sys.path.append('/caffe/python')
import caffe
import cv2
import numpy as np
import progressbar as pb
sys.path.append('../')
from common import functions as fs

#========================== To be configure ====================
TXT = '/home/eryuan/WIDER_train/wider_face_train.txt'  # label file
BASE_PATH = 'data_images'   # image base path, image path = image base path + label path
MEG_PATH = BASE_PATH + '/neg' # negative images path
POS_PATH = BASE_PATH + '/pos' # postive images path
PART_PATH = BASE_PATH + '/part' # calibration images path
deploy = '../Fstage/models/net.prototxt'
caffemodel = '../Fstage/models/solver_iter_150000.caffemodel'
#===============================================================

GPU_ID = 0  # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
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
    h, w, c = img.shape
    scales = scaleImage(img, minsize)
    out = []
    rects = []
    for i, s in enumerate(scales):
        f = 0.8 ** i
        h_, w_, c_ = s.shape
        # print s.shape
        net_caffe.blobs['X'].reshape(1, 3, h_, w_)
        img_ = (s - 127.5) / 127.5
        img__ = img_.transpose((2, 0, 1))
        net_caffe.blobs['X'].data[...] = img__
        caffe.set_device(GPU_ID)
        out_ = net_caffe.forward()
        c__, h__, w__ = out_['prob'][0].shape
        for (y, x), pscore in np.ndenumerate(out_['prob'][0][1]):
            # for x in range(w__):
            #     for y in range(h__):
            if pscore > 0.5:
                x1 = x * strip
                y1 = y * strip
                x2 = x1 + minsize
                y2 = y1 + minsize
                minw = int(minsize * (1.0 / f))
                x1 = max(0, x1 * (1.0 / f))
                y1 = max(0, y1 * (1.0 / f))
                x2 = min(img.shape[1], x2 * (1.0 / f))
                y2 = min(img.shape[0], y2 * (1.0 / f))
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                if x2 > x1 and y2 > y1:
                    rects.append([x1, y1, x2, y2, out_['prob'][0][1][y][x],
                                  (x2 - x1) * out_['conv4-2'][0][0][y][x],
                                  (y2 - y1) * out_['conv4-2'][0][1][y][x],
                                  (x2 - x1) * out_['conv4-2'][0][2][y][x],
                                  (y2 - y1) * out_['conv4-2'][0][3][y][x]
                                  ]
                                 )

    rects = NMS(rects, 0.8)
    rects = np.array(rects, dtype=float)
    rects[:, :4] += rects[:, 5:]
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


def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def IoUs(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


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

        rects = detectFaceInput(img_)
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
