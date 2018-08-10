#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:00 AM
# @Author  : Ject.Y
# @Site    : 
# @File    : pythonLayer.py
# @Software: PyCharm
# @Copyright: BSD2.0
# @Function : function
# How to run : run

# for the MTCNN caffe pythonLayer
import os
import sys
sys.path.append('caffe/python')  # caffe python path
import caffe
import numpy as np


class bridge(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        label = bottom[1].data
        self.valid_index = np.where(label != -1)[0]
        self.valid_count = len(self.valid_index)
        top[0].reshape(self.valid_count,2,1,1)
        top[1].reshape(self.valid_count,1)
    def forward(self, bottom, top):
        top[0].data[...][...] = 0.0
        top[1].data[...][...] = 0.0
        top[0].data[...] = bottom[0].data[self.valid_index]
        top[1].data[...] = bottom[1].data[self.valid_index]
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0] and self.valid_count > 0:
            bottom[0].diff[...] = 0
            bottom[0].diff[self.valid_index] = top[0].diff[...]
        if propagate_down[1] and self.valid_count > 0:
            bottom[1].diff[...] = 0
            bottom[1].diff[...] = top[1].diff[...]


class bridge_fc(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        label = bottom[1].data
        self.valid_index = np.where(label != -1)[0]
        self.valid_count = len(self.valid_index)
        top[0].reshape(self.valid_count,2)
        top[1].reshape(self.valid_count,1)
    def forward(self, bottom, top):
        top[0].data[...][...] = 0.0
        top[1].data[...][...] = 0.0
        top[0].data[...] = bottom[0].data[self.valid_index]
        top[1].data[...] = bottom[1].data[self.valid_index]
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0] and self.valid_count > 0:
            bottom[0].diff[...] = 0
            bottom[0].diff[self.valid_index] = top[0].diff[...]
        if propagate_down[1] and self.valid_count > 0:
            bottom[1].diff[...] = 0
            bottom[1].diff[...] = top[1].diff[...]


class regression_Layer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        roi = bottom[1].data
        self.valid_index = np.where(roi[:, 0] != -1)[0]
        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            self.diff[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data).reshape(bottom[0].data.shape)[self.valid_index]
            top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i] or self.N == 0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
