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
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return prepare_data(img)


def resize(img_data) -> np.ndarray:
    '''

    :param img_data: 图像矩阵
    :return: 28*28的图像
    '''
    target_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    if img_data.shape == target_shape:
        return img_data
    ret = cv2.resize(img_data, target_shape, interpolation=cv2.INTER_LINEAR)
    return ret


def prepare_data(img_data):
    data = resize(img_data).astype(np.float32)
    data = (data - (config.PIXEL_DEPTH / 2.0)) / config.PIXEL_DEPTH
    data = data.reshape(1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS)
    return data
