# -*- coding: utf-8 -*-
# @Time    : 2018/3/25 下午3:41
# @Author  : pzque

import mh_config as config
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_one(img_path):
    '''

    :param img_path:
    :return:
    '''
    data = np.ndarray(shape=[1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS])
    img = cv2.imread('samples/2.jpg', cv2.IMREAD_GRAYSCALE)
    img = resize(img)
    data[0, :, :, 0] = img
    return data


def resize(img_data):
    '''

    :param img_data: 图像矩阵
    :return: 28*28的图像
    '''

    return cv2.resize(img_data, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
