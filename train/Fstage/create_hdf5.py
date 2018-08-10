#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 10:42 AM
# @Author  : Ject.Y
# @Site    :
# @File    : create_h5_cls.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : Generate first stage training and validation HDF5 data
# How to run : run

import h5py, os
import numpy as np
import random
import cv2


# ============================= To be configure ==================================
POS_TXT = 'pos.txt'   # postive file path
NEG_TXT = 'neg.txt'   # negative file path
ROI_TXT = 'part.txt'  # calibration file path
HDF5_SAVE_PATH = 'HDF5'           # save path
SIZE = 24  # fixed size to all images
image_base_path = ''  # image base path
counts = 2   #the number of train data files
counts2 = 2  #the number of validation data files
#==================================================================================

# read the pos txt
with open(POS_TXT, 'r') as T:
    pos_lines = T.readlines()
# read neg txt
with open(NEG_TXT, 'r') as T_:
    neg_lines = T_.readlines()

# read part txt
with open(ROI_TXT, 'r') as T_:
    part_lines = T_.readlines()

lines = []
pos_count = len(pos_lines)
neg_count = len(neg_lines)
part_count = len(part_lines)
if neg_count > (pos_count * 3):
    neg_count = pos_count * 3
if part_count > pos_count:
    part_count = pos_count
neg_lines = neg_lines[:neg_count]
part_lines = part_lines[:part_count]
lines.extend(pos_lines)
lines.extend(neg_lines)
lines.extend(part_lines)

# get the train and validate data size
random.shuffle(lines)
random.shuffle(lines)
random.shuffle(lines)
random.shuffle(lines)
random.shuffle(lines)
random.shuffle(lines)

# cout the training and validaition data
train_data = lines[:int((len(lines) * 0.8))]
val_data = lines[len(train_data):]

f_train = open('train_h5.txt','w')
f_val = open('val_h5.txt','w')


signCout = int(len(train_data) / float(counts))
numCount = 0
while numCount < counts:
    startCount = numCount * signCout
    endCount = startCount + signCout
    if numCount == (counts - 1):
        endCount = len(train_data)
    train_datas = train_data[startCount:endCount]
    train_X = np.zeros((len(train_datas), 3, SIZE, SIZE), dtype='f4')
    train_y = np.zeros((len(train_datas), 1), dtype='f4')
    train_r = np.zeros((len(train_datas), 4), dtype='f4')
    for i, l in enumerate(train_datas):
        if i % 1000 == 0:
            print "Processing %dth image of training dataset  %d" % ((i + 1), len(train_datas))
        sp = l.strip().split(' ')
        img = cv2.imread(
                image_base_path + sp[0])
        img = cv2.resize(img,(SIZE,SIZE))
        img = (img - 127.5) / 127.5
        transposed_img = img.transpose((2, 0, 1))
        train_X[i] = transposed_img
        train_y[i] = float(sp[1])
        train_r[i] = [-1,-1,-1,-1]
        if len(sp) > 2:
            train_r[i] = sp[2:]




    print "Generate the training  HDF5 images Dataset..."

    with h5py.File(HDF5_SAVE_PATH + '/train%d.h5'%numCount, 'w') as H:
        H.create_dataset('X', data=train_X)  # note the name X given to the dataset!
        H.create_dataset('y', data=train_y)  # note the name y given to the dataset!
        H.create_dataset('r', data=train_r)  # note the name r given to the dataset!
    f_train.write(HDF5_SAVE_PATH + '/train%d.h5\n'%numCount)
    numCount += 1

signCout = int(len(val_data) / float(counts2))
numCount = 0
while numCount < counts2:
    startCount = numCount * signCout
    endCount = startCount + signCout
    if numCount == (counts2 - 1):
        endCount = len(val_data)
    val_datas = val_data[startCount:endCount]

    val_X = np.zeros((len(val_datas), 3, SIZE, SIZE), dtype='f4')
    val_y = np.zeros((len(val_datas), 1), dtype='f4')
    val_r = np.zeros((len(val_datas), 4), dtype='f4')
    for i, l in enumerate(val_datas):
        if i % 1000 == 0:
            print "Processing %dth image of validata dataset  %d" % ((i + 1), len(val_datas))
        sp = l.strip().split(' ')
        img = cv2.imread(
               image_base_path + sp[0])
        img = cv2.resize(img, (SIZE, SIZE))
        img = (img - 127.5) / 127.5
        transposed_img = img.transpose((2, 0, 1))
        val_X[i] = transposed_img
        val_y[i] = float(sp[1])
        val_r[i] = [-1,-1,-1,-1]
        if len(sp) > 2:
            val_r[i] = sp[2:]
    print "Generate the validation  HDF5 images Dataset..."

    with h5py.File(HDF5_SAVE_PATH + '/validation%d.h5' % numCount, 'w') as H:
        H.create_dataset('X', data=val_X)  # note the name X given to the dataset!
        H.create_dataset('y', data=val_y)  # note the name y given to the dataset!
        H.create_dataset('r', data=val_r)  # note the name r given to the dataset!

    f_val.write(HDF5_SAVE_PATH + '/validation%d.h5\n' % numCount)
    numCount += 1

print 'Done'
quit()
