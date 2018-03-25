# -*- coding: utf-8 -*-
# @Time    : 2018/3/25 下午5:26
# @Author  : pzque


import cv2
import numpy as np

import preprocessing
from inference import Inferencer


class InteractiveSession(object):
    def __init__(self):
        self.infer = Inferencer()
        self.infer.open()

        # 是否正在绘图
        self.drawing = False
        self.lastx, self.lasty = -1, -1
        self.to_reset_drawing_area = False
        self.convas = np.zeros((280, 570, 1), np.uint8)
        self.convas[:, 280:290] = 255

        # 左半部分是绘图区域
        self.drawing_area = self.convas[:, :280]
        # 有半部分展示结果
        self.result_area = self.convas[:, 290:]

    def reset_drawing_area(self):
        '''
        将绘图区域重置为全黑
        :return:
        '''
        self.drawing_area[:] = 0
        self.to_reset_drawing_area = False

    def reset_result_area(self):
        '''
        将结果区域重置为全黑
        :return:
        '''
        self.result_area[:] = 0

    def draw_num(self, num):
        '''
        在结果展示区显示一个数字
        :param num:
        :return:
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.result_area, str(num), (55, 220), font, 8, (255, 255, 255), 15)

    def recognize_and_display_result(self):
        '''
        识别左边用户画好的图并将结果显示在右边的区域
        :return:
        '''
        self.reset_result_area()
        data = preprocessing.prepare_data(self.drawing_area)
        num = self.infer.inference_once(data)
        self.draw_num(num)
        # 结果展示完成后需要重置绘图区域以等待用户下一次绘图
        self.to_reset_drawing_area = True

    def run(self):
        '''
        运行交互会话
        :return:
        '''
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mouse_callback)

        while (True):
            cv2.imshow('image', self.convas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                self.recognize_and_display_result()
            elif key == ord('q'):
                break
            elif key == 27:
                break

        self.infer.close()
        cv2.destroyAllWindows()

    # mouse callback function
    def mouse_callback(self, event, x, y, flags, param):
        '''
        鼠标回调函数
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            # 如果用户在右边单击
            if x > 290:
                self.recognize_and_display_result()
                return
            # 如果用户在左边单击, 那么是绘图事件
            if x < 280:
                if self.to_reset_drawing_area:
                    self.reset_drawing_area()

                self.drawing = True
                self.lastx, self.lasty = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.drawing_area, (self.lastx, self.lasty), (x, y), 255, 20)
            self.lastx, self.lasty = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False


def main():
    sess = InteractiveSession()
    sess.run()


if __name__ == '__main__':
    main()
