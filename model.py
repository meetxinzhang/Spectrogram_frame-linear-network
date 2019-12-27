# coding: utf-8
# ---
# @File: model.py
# @description: 模型类
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月18, 2019
# ---


import tensorflow as tf
from PIL import Image
import scipy.misc
import os
from linear_3d_layer import Linear3DLayer


class Model_X(tf.keras.Model):
    """
    继承自基类 tf.keras.Model
    """
    def __init__(self, rnn_units, num_class):
        super(Model_X, self).__init__()
        self.rnn_units = rnn_units
        self.num_class = num_class
        self.i = 0

        # 线性层
        self.lcl1 = Linear3DLayer(filters=8, kernel_size=[1, 3, 75, 6],
                                  activate_size=[3, 1, 2], activate_stride=[3, 1, 1])
        self.lcl2 = Linear3DLayer(filters=8, kernel_size=[8, 3, 36, 3],
                                  activate_size=[3, 1, 2], activate_stride=[3, 1, 1])
        self.lcl3 = Linear3DLayer(filters=8, kernel_size=[8, 3, 17, 2],
                                  activate_size=[3, 1, 1], activate_stride=[3, 1, 1])

        # 池化层
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_first')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_first')
        self.pooling3 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_first')

        # 3DCNN
        # self.conv3d1 = tf.keras.layers.Conv3D(filters=32, kernel_size=[3, 78, 6], strides=[1, 1, 6],
        #                                       use_bias=True,
        #                                       activation=tf.nn.leaky_relu, padding='same',
        #                                       kernel_initializer=tf.keras.initializers.he_normal(),
        #                                       bias_initializer=tf.zeros_initializer(),
        #                                       data_format='channels_first')

        # self.conv3d2 = tf.keras.layers.Conv3D(filters=16, kernel_size=[3, 38, 3], strides=[1, 1, 3],
        #                                       use_bias=True,
        #                                       activation=tf.nn.leaky_relu, padding='same',
        #                                       kernel_initializer=tf.keras.initializers.he_normal(),
        #                                       bias_initializer=tf.zeros_initializer(),
        #                                       data_format='channels_first')

        # self.conv3d3 = tf.keras.layers.Conv3D(filters=8, kernel_size=[3, 19, 2], strides=[1, 1, 2],
        #                                       use_bias=True,
        #                                       activation=tf.nn.leaky_relu, padding='same',
        #                                       kernel_initializer=tf.keras.initializers.he_normal(),
        #                                       bias_initializer=tf.zeros_initializer(),
        #                                       data_format='channels_first')

        # GRU 网络
        self.cell1 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True)
        self.cell2 = tf.keras.layers.CuDNNGRU(units=self.num_class)

        # BatchNormal
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        # self.pooling_a = tf.keras.layers.AveragePooling2D(pool_size=[1, 1, 2], strides=[1, 1, 2], padding='same',
        #                                                   data_format='channels_first')

        # drop = tf.keras.layers.Dropout(rate=drop_rate)
        # FC
        # self.fla = tf.keras.layers.Flatten(data_format='channels_last')
        # self.fc1 = tf.keras.layers.Dense(units=128, use_bias=True, activation=None,
        #                                  kernel_initializer=tf.keras.initializers.he_normal(),
        #                                  bias_initializer=tf.constant_initializer)
        # self.fc2 = tf.keras.layers.Dense(units=num_class, use_bias=True, activation=None,
        #                                  kernel_initializer=tf.keras.initializers.he_normal(),
        #                                  bias_initializer=tf.constant_initializer)

    def call(self, inputs, drop_rate=0.3, **kwargs):
        """
        组织了三层时频帧线性层，两层GRU，然后输出GRU的最后一个时间状态作为logits，其中串联了 BatchNormal
        :param drop_rate: Dropout的比例=0.3，这个超参数没用到
        :param inputs: [?, 1, 200, 80, 4]
        :return: logits
        """
        is_training = tf.equal(drop_rate, 0.3)
        # print('inputs ', np.shape(inputs))

        lc1 = self.lcl1(inputs)
        # print('conv1: ', sc1.get_shape().as_list())
        lc1 = self.bn1(lc1, training=is_training)
        pool1 = self.pooling1(lc1)  # (?, filters, 99, 39, 4)
        # print('pool1: ', pool1.get_shape().as_list())

        lc2 = self.lcl2(pool1)
        lc2 = self.bn2(lc2, training=is_training)
        pool2 = self.pooling2(lc2)  # (?, filters, 49, 19, 2)
        # print('pool2: ', pool2.get_shape().as_list())

        lc3 = self.lcl3(pool2)
        lc3 = self.bn3(lc3, training=is_training)
        pool3 = self.pooling3(lc3)  # (?, filters, 24, 9, 1)
        # pool3 = self.pooling_a(pool3)
        pool3 = tf.squeeze(pool3, axis=-1)  # [?, filters, 24, 9]
        # print('pool3: ', pool3.get_shape().as_list())

        # x_rnn = tf.squeeze(pool3, axis=2)  # (?, 8, 2, 10, 5)
        # x_rnns = tf.unstack(pool3, axis=2)  # 展开帧维度  2*[?, 8, 10, 5]
        # x_rnn = tf.concat(x_rnns, axis=3)  # 合并到行维度  [?, 8, 10, 10]

        if not is_training:
            # self.draw_hid_features(inputs, pool3)
            pass
        ##################################################################
        x_rnns = tf.unstack(pool3, axis=1)  # 展开通道维度  filters*[?, 17, 10]
        x_rnn = tf.concat(x_rnns, axis=2)  # 合并到列维度  [?, 17, filters*10=80]

        # x_rnn = tf.transpose(x_rnn, [0, 2, 1])  # [?, 10, 80]

        # rnn_output = []
        # for i in range(self.num_class):
        #     name = "ltsm_" + str(i)
        #     cell = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, name=name)
        #     fc = tf.keras.layers.Dense(units=1, use_bias=True, activation=None,
        #                                kernel_initializer=tf.keras.initializers.he_normal(),
        #                                bias_initializer=tf.constant_initializer())
        #     drop = tf.keras.layers.Dropout(rate=drop_rate)
        #
        #     item_out = cell(inputs=x_rnn)  # [?, 64]
        #     fc_out = drop(item_out)
        #     fc_out2 = fc(fc_out)  # [?, 1]
        #     cell = None
        #     drop = None
        #     fc = None
        #
        #     rnn_output.append(fc_out2)  # [4, ?, 1]
        # rnn_output = tf.squeeze(rnn_output)  # [4, ?]
        # logits = tf.transpose(rnn_output)  # [?, 4]
        ####################################################################
        # rnn_output = []
        # for _index in range(4):
        #     name = "gru_" + str(_index)
        #     cell = tf.keras.layers.CuDNNLSTM(units=32, name=name)
        #     item_out = cell(inputs=x_rnns[_index])  # [?, 25, rnn_units]
        #     cell = None
        #
        #     rnn_output.append(item_out)
        #
        # output = tf.concat(rnn_output, 1)  # [?, self.rnn_units*4]
        # drop = tf.keras.layers.Dropout(rate=drop_rate)(output)
        # logits = self.fc2(drop)
        ####################################################################
        # drop = tf.keras.layers.Dropout(rate=drop_rate)
        # fla = self.fla(x_rnn)
        # fc1 = self.fc1(fla)
        # fc1 = drop(fc1)
        # logits = self.fc2(fc1)
        ####################################################################

        cell_out1 = self.cell1(x_rnn)
        logits = self.cell2(cell_out1)

        return logits

    def draw_hid_features(self, inputs, batch):
        """
        绘制中间层的特征图，保存在本地/hid_pic，第120-121行调用
        :param inputs: [?, 1, 100, 80, 6]
        :param batch: [?, 8, 13, 10]
        """
        import numpy
        inputs = numpy.squeeze(inputs)  # [?, 100, 80, 6]
        batch = batch.numpy()

        index_sample = 0
        for sample in batch:
            # [8, 13, 10]
            index_channel = 0

            yuan_tus = inputs[index_sample]
            yuan_tu = numpy.hstack(yuan_tus)

            save_dir = 'hid_pic' + '/batch_' + str(self.i) + '/' + str(index_sample)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Image.fromarray(yuan_tu).convert('RGB').save(save_dir + '/' + 'yuan_tu.jpg')

            for feature in sample:
                # [13, 10]
                save_path = save_dir + '/' + str(index_channel) + '.jpg'
                scipy.misc.imsave(save_path, feature.T)
                # Image.fromarray(feature).convert('RGB').save(save_path)

                index_channel += 1
            index_sample += 1

        self.i += 1

############
# 8 16 8
