import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network

# 定义模型存储的位置
CKPT_DIR = 'ckpt'

class Train:
    def __init__(self):
        self.net = Network()

        # 初始化session
        # Network()只是构造了一张计算图，计算要放到会话(session)中
        self.sess = tf.Session()
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 初始化模型保存
        self.saver = tf.train.Saver()

        # 读取训练和测试数据
        self.data = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    def train(self):
        # batch_size是指每次迭代训练，传入训练的图片张数
        batch_size = 100
        # 总的训练次数
        train_step = 3000

        # 数据可视化
        # merge所有的summary node
        merged_summary_op = tf.summary.merge_all()
        # 可视化存储目录为当前文件夹下的 log
        merged_writer = tf.summary.FileWriter("./log", self.sess.graph)

        # 开始训练
        for i in range(train_step):
            # 从数据集中获取输入和标签
            x, label = self.data.train.next_batch(batch_size)
            # 每次计算train，更新整个网络
            # loss仅是为了看到损失的大小，方便打印
            _, loss, merged_summary = self.sess.run([self.net.train, self.net.loss, merged_summary_op],
                                 feed_dict={self.net.x: x, self.net.label: label, self.net.keep_prob: 0.75})
            # loss打印
            if (i + 1) % 10 == 0:
                print('第%5d步，当前loss：%.2f' % (i + 1, loss))

            # log记录
            if i % 100 == 0:
                merged_writer.add_summary(merged_summary, i)

        # 模型存储
        self.saver.save(self.sess, CKPT_DIR + '/model')


    def calculate_accuracy(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels
        # 仅计算accuracy这个张量，不会更新网络
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: test_x, self.net.label: test_label, self.net.keep_prob: 0.75})
        print('准确率：%.2f，共测试了%d涨照片' % (accuracy, len(test_label)))


if __name__ == '__main__':
    app = Train()
    app.train()
    app.calculate_accuracy()