# disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

class Network:
    def __init__(self):
        # 学习速率
        self.learning_rate = 0.3
        # 训练步数
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # 定义算法公式(包含隐含层)
        input_units = 784   # 输入向量维度(数据)
        hidden_units = 300  # 隐含层向量维度
        output_units = 10   # 输出向量维度(标签)
        w1 = tf.Variable(tf.truncated_normal([input_units, hidden_units], stddev=0.1), name='w1')  # 权重w1
        b1 = tf.Variable(tf.zeros(hidden_units), name='b1')    # 隐含层偏置项
        w2 = tf.Variable(tf.zeros([hidden_units, output_units]), name='w2')    # 权重w2
        b2 = tf.Variable(tf.zeros([output_units]), name='b2')  # 输出层偏置项

        # 输入张量
        self.x = tf.placeholder(tf.float32, [None, input_units], name='x')
        self.keep_prob = tf.placeholder(tf.float32)  # Dropout(保留节点的比率)(防止过拟合的策略)

        # 标签值
        self.label = tf.placeholder(tf.float32, [None, output_units], name='label')

        # 定义隐含层(激活函数为RELU)
        hidden_layer = tf.nn.relu(tf.matmul(self.x, w1) + b1)
        hidden_layer_dropout = tf.nn.dropout(hidden_layer, self.keep_prob)   # Dropout的功能，keep_prob参数即为保留数据不置0的比例

        # 输出
        self.y = tf.nn.softmax(tf.matmul(hidden_layer_dropout, w2) + b2, name='y')    # 模型训练预测输出

        # 损失函数(交叉熵)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.y), reduction_indices=[1]), name='loss')

        # 训练(使损失函数逐步减小)
        self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        # 验证准确率
        predict = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))

        # 数据可视化
        # 直方图
        tf.summary.histogram('w1', w1)
        tf.summary.histogram('b1', b1)
        tf.summary.histogram('w2', w2)
        tf.summary.histogram('b2', b2)
        # 标量图
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    # 简单模型训练示例(不包含隐含层)
    def demo(self):
        # 学习速率，一般在0.00001 - 0.5之间
        self.learning_rate = 0.001

        # 输入张量28 * 28 = 784个像素的图片一维向量
        self.x = tf.placeholder(tf.float32, [None, 784])

        # 标签值 one-hot编码
        self.label = tf.placeholder(tf.float32, [None, 10])

        # 权重，初始化为全0
        self.w = tf.Variable(tf.zeros([784, 10]))
        # 偏置bias，初始化为全0
        self.b = tf.Variable(tf.zeros([10]))
        # 输出y = softmax(x * w + b)
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

        # 损失，即交叉熵
        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

        # 记录已经训练的次数
        self.global_step = tf.Variable(0, trainable=False)

        # 反向传播、采用梯度下降的方法。调整w与b，使得损失(loss)最小
        # loss越小，计算出来的y值和标签(label)值越接近，准确率越高
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

        # 验证准确率
        # argmax返回最大值的下标，最大值的下标即为答案
        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
        # reduce_mean即求predict的平均数，即正确个数 / 总数,即正确率
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))
