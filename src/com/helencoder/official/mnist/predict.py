# disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
from PIL import Image

from model import Network

class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        CKPT_DIR = 'ckpt'
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存任何模型")

    # 照片格式转换为mnist格式
    def img2mnist(self, image_path):

        # 读取图片转成灰度格式
        img = Image.open(image_path).convert('L')

        # resize的过程
        if img.size[0] != 28 or img.size[1] != 28:
            img = img.resize((28, 28))

        # 暂存像素值的一维数组
        arr = []
        for i in range(28):
            for j in range(28):
                # mnist 里的颜色是0代表白色（背景），1.0代表黑色
                pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
                # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
                arr.append(pixel)

        x = np.array(arr).reshape(1, 784)
        return x

    def predict(self, image_path):
        x = self.img2mnist(image_path)
        y = self.sess.run([self.net.y], feed_dict={self.net.x: x, self.net.keep_prob: 0.75})

        # 只传入单张照片，取y[0]即可
        print(image_path + '    -> Predict ', np.argmax(y[0]))


if __name__ == '__main__':
    app = Predict()
    app.predict('./test_imgs/example3.png')
    app.predict('./test_imgs/example5.png')
    app.predict('./test_imgs/0.png')
    app.predict('./test_imgs/1.png')
    app.predict('./test_imgs/4.png')