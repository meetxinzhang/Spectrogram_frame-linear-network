import tensorflow as tf
import numpy as np


class Linear3DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activate_size, activate_stride):
        """
        该层我起名叫时频帧线性层，其中特殊构造的过滤器类似于在流水线上工作的工人（论文中我说是像一个带通滤波器 band-pass filter），
        过滤器只在连续帧的时间维度上移动，适合处理时序数据（视频，音频等）
        参数个数： filters * kernel_size
        :param filters: 过滤器的个数
        :param cache: 上下缓冲区宽度
        :param kernel_size: 过滤器的大小，维度说明： [上一层通道数，连续帧深度，单帧高，单帧宽]
        :param activate_size: 在过滤器上的激活窗口大小，维度说明： [连续帧深度，单帧高，单帧宽]
        :param activate_stride: 在过滤器上的激活窗口移动步长，维度说明： [连续帧深度，单帧高，单帧宽]
        """
        super(Linear3DLayer, self).__init__()
        self.filters = filters
        self.margin = 0
        [self.c, self.d, self.h, self.w] = kernel_size

        self.weight = tf.get_variable('weight', shape=[self.filters, 1, self.c, self.d, self.h, self.w],
                                      initializer=tf.keras.initializers.he_normal())
        self.bias = tf.get_variable('bias', shape=[self.filters, 1, self.c, self.d, self.h, self.w],
                                    initializer=tf.keras.initializers.he_normal())
        self.conv3d = tf.keras.layers.Conv3D(filters=1, kernel_size=activate_size, strides=activate_stride,
                                             use_bias=False,
                                             activation=tf.nn.leaky_relu, padding='same',
                                             kernel_initializer=tf.ones_initializer(),
                                             data_format='channels_first',
                                             name='activate_conv',
                                             trainable=False)

    def __multi_granularity_activate__on_kernel__(self, inputs):
        """
        tf 的卷积函数，这部分参数固定为1，不参与训练，只是为了利用卷积运算的功能，在过滤器的3x3块上进行多粒度扫描，应用激活函数，并合并通道
        :param inputs: [?, c, 3, h, w]
        :return: [?, c, 1, h, w]
        """
        out_map = self.conv3d(inputs)

        return out_map

    def __margin_multiply__(self, d_slice, w_tiled_slice, b_tiled_slice):
        """
        :param d_slice: [bs, c, d, h, w]
        :param w_tiled_slice: [bs, c, d, h, w]
        :return: [bs, c, d, h, w]
        """
        for i in range(self.margin):
            h_slice = tf.slice(d_slice, [0, 0, 0, i, 0], [-1, -1, -1, self.h, -1])
            b_tiled_slice = tf.multiply(h_slice, w_tiled_slice) + b_tiled_slice

        return tf.multiply(b_tiled_slice, 1/4)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [?, channel, depth, h, w]
        :param training: None
        :param mask: None
        :return: features_map has a same shape to inputs
        """
        [batch_size, channel, depth, height, _] = np.shape(inputs)
        assert channel == self.c & height > self.h

        self.margin = height-self.h

        inputs = tf.cast(inputs, tf.float32)

        w_tiled = tf.tile(self.weight, [1, batch_size, 1, 1, 1, 1], name="w_tiled")  # [f, bs, c, d, h, w]
        b_tiled = tf.tile(self.bias, [1, batch_size, 1, 1, 1, 1], name="b_tiled")  # [f, bs, c, d, h, w]

        features_map = tf.zeros(shape=[batch_size, self.filters, 0, self.h, self.w])

        # 逐帧扫描
        for i in range(depth):
            try:
                d_slice = tf.slice(inputs, [0, 0, i, 0, 0], [-1, -1, self.d, -1, -1])  # [bs, c, d, h, w]
            except tf.errors.InvalidArgumentError:
                # 到达了序列末尾
                # print('InvalidArgumentError: ', i)
                # if i+self.d == depth+1:
                #     # 在这里进行 padding
                continue

            slice_map = tf.zeros(shape=[batch_size, 0, 1, self.h, self.w])

            for f in range(self.filters):
                w_tiled_slice = w_tiled[f]
                b_tiled_slice = b_tiled[f]

                a = self.__margin_multiply__(d_slice, w_tiled_slice, b_tiled_slice)  # [bs, c, d, h, w]
                b = self.__multi_granularity_activate__on_kernel__(a)  # 利用卷积层合并通道 [bs, 1?, 1, h, w]

                slice_map = tf.concat([slice_map, b], axis=1)  # 组合为多通道 区块特征图 [bs, filter, 1, h, w]

            features_map = tf.concat([features_map, slice_map], axis=2)  # 组合为多帧（完整的）特征图

            # 释放内存
            d_slice = None
            slice_map = None
            a = None
            b = None

        return features_map
