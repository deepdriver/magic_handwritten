# magic_handwritten

Handwritten digits and letters recognition with deep learning.

## 依赖
```
tensorflow >= 1.4
opencv-python
```

## 如何使用
克隆本仓库, 安装好必须依赖, 在项目目录执行: 
```
python main.py
```
界面如图:
![windows.png](https://github.com/deepdriver/magic_handwritten/raw/master/docs/images/window.png)

在左边绘图, 鼠标左键单击右边或者按空格键显示识别结果, 按'q'退出程序.
## 项目结构
- 'train.py' 构建了一个简单的CNN网络, 然后使用mnist数据集训练, 最后在测试集上达到0.8%的错误率. 我们已经训练并保存了模型, 所以不需要再训练.
- 'checkpoints/' 模型保存在这里
- 'inference.py' 加载模型识别图像的代码在这里面


