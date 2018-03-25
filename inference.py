# -*- coding: utf-8 -*-
# @Time    : 2018/3/25 下午3:28
# @Author  : pzque

import tensorflow as tf
import mh_config as config
import numpy as np


class Inferencer(object):
    def __init__(self):
        self.graph = None
        self.sess = None
        self.eval_batch = None
        self.eval_once = None
        self.eval_data_batch = None
        self.eval_data_once = None

    def open(self):
        '''
        加载模型
        '''
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('checkpoints/model.meta')
        saver.restore(self.sess, 'checkpoints/model')

        graph = tf.get_default_graph()

        self.eval_batch = tf.get_collection('eval_batch')[0]
        self.eval_once = tf.get_collection('eval_batch')[0]

        self.eval_data_batch = graph.get_operation_by_name('eval_data_batch').outputs[0]
        self.eval_data_once = graph.get_operation_by_name('eval_data_once').outputs[0]

    def close(self):
        self.sess.close()

    def inference_once(self, data: np.ndarray):
        shape = (1, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        msg = "The input data must be a numpy.ndarray of shape {shape}.".format(shape=shape)
        if type(data) != np.ndarray or data.shape != shape:
            raise Exception(msg)

        result = self.sess.run(self.eval_once, feed_dict={self.eval_data_once: data})
        return result

    def inference_batch(self, data: np.ndarray):
        shape = (config.EVAL_BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        msg = "The input data must be a numpy.ndarray of shape {shape}.".format(shape=shape)
        if type(data) != np.ndarray or data.shape != shape:
            raise Exception(msg)
        result = self.sess.run(self.eval_batch, feed_dict={self.eval_data_batch: data})
        return result

    def get_batch_size(self):
        return config.EVAL_BATCH_SIZE


def main():
    i = Inferencer()
    i.open()
    i.inference_once(1)
    i.inference_batch(2)
    i.close()


if __name__ == "__main__":
    main()
