# -*- coding: utf-8 -*-
# @Time    : 2018/3/25 下午5:26
# @Author  : pzque


import cv2
import numpy as np
import matplotlib.pylab as plt

import preprocessing
from inference import Inferencer

infer = Inferencer()
infer.open()

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
reset = False
img = np.zeros((280, 280, 1), np.uint8)


# mouse callback function
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, reset, img

    if event == cv2.EVENT_LBUTTONDOWN:
        if reset:
            img[:] = 0
            reset = False
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x, y), 255, 20)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def main():
    global reset
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_line)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            # cv2.imwrite('test.jpg', img)
            data = preprocessing.prepare_data(img)
            num = infer.inference_once(data)
            reset = True
            print(num)
        elif k == ord('q'):
            break
        elif k == 27:
            break

    cv2.destroyAllWindows()
    infer.close()


if __name__ == '__main__':
    main()
