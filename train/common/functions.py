#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 9:45 AM
# @Author  : Ject.Y
# @Site    : 
# @File    : functions.py
# @Software: PyCharm
# @Copyright: BSD 2.0
# @Function : function
# How to run : run

import os
import sys
import numpy as np

def dirlist(path, allfile,sufix = '*'):
    """
        as the os.listdir is a function to list all the files in a floder
        but it will list all the hidden files, too
        but most of time we just need the no-hidden files
        this function is just to do this thing
        list no-hidden files in a path
        :param path: the path
        :return: the files whiout none hidden
        """

    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if filepath.startswith('.'):
            continue
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            if not sufix == '*':
                if os.path.splitext(filepath)[1] == ('.'+sufix):
                    allfile.append(filepath)
            else:
                allfile.append(filepath)
    return allfile


def dirlist2(path, allfile,sufix = ['*']):
    """
        as the os.listdir is a function to list all the files in a floder
        but it will list all the hidden files, too
        but most of time we just need the no-hidden files
        this function is just to do this thing
        list no-hidden files in a path
        :param path: the path
        :return: the files whiout none hidden
        """

    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if filepath.startswith('.'):
            continue
        if os.path.isdir(filepath):
            dirlist2(filepath, allfile,sufix)
        else:
            if '*' not in sufix:
                if os.path.splitext(filepath)[1] in sufix:
                    allfile.append(filepath)
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
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

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


def mkdirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d/%d' % ("#"*rate_num, " "*(100-rate_num), num,total )
    sys.stdout.write(r)
    sys.stdout.flush()


