# -*- coding: utf-8 -*-
# @Time    : 2018/3/25 下午3:28
# @Author  : pzque

import tensorflow as tf
import mh_config as config
import numpy as np
import preprocessing


class Inferencer(object):
    def __init__(self):
        self.__graph = None
        self.__sess = None
        self.__eval_batch = None
        self.__eval_once = None
        self.__eval_data_batch = None
        self.__eval_data_once = None

    def open(self):
        '''
        加载模型
        '''
        self.__sess = tf.Session()
        saver = tf.train.import_meta_graph('checkpoints/model.meta')
        saver.restore(self.__sess, 'checkpoints/model')

        graph = tf.get_default_graph()

        self.__eval_batch = tf.get_collection('eval_batch')[0]
        self.__eval_once = tf.get_collection('eval_once')[0]

        self.__eval_data_batch = graph.get_operation_by_name('eval_data_batch').outputs[0]
        self.__eval_data_once = graph.get_operation_by_name('eval_data_once').outputs[0]

    def close(self):
        self.__sess.close()

    def __inference_once(self, data: np.ndarray):
        shape = (1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS)
        msg = "The input data must be a numpy.ndarray of shape {shape}.".format(shape=shape)
        if type(data) != np.ndarray or data.shape != shape:
            raise Exception(msg)

        result = self.__sess.run(self.__eval_once, feed_dict={self.__eval_data_once: data})
        return result

    def __inference_batch(self, data: np.ndarray):
        shape = (config.EVAL_BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS)
        msg = "The input data must be a numpy.ndarray of shape {shape}.".format(shape=shape)
        if type(data) != np.ndarray or data.shape != shape:
            raise Exception(msg)
        result = self.__sess.run(self.__eval_batch, feed_dict={self.__eval_data_batch: data})
        return result

    def inference_img(self, img_path):
        data = preprocessing.load_one(img_path)
        softmax_array = self.__inference_once(data)
        return np.argmax(softmax_array)

    def get_batch_size(self):
        return config.EVAL_BATCH_SIZE


def main():
    infer = Inferencer()
    infer.open()
    img_path = 'samples/2.jpg'
    num = infer.inference_img(img_path)
    print(num)
    infer.close()


if __name__ == "__main__":
    main()
