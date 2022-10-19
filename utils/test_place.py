# coding: utf-8
# ---
# @File: test_place.py
# @description:
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 9月5, 2019
# ---


import numpy as np
# import tensorflow as tf
#
# tf.enable_eager_execution()
#
# x = [[[0, 2.1],
#       [0, 2.1]],
#
#      [[1, 1],
#       [1, 1]],
#
#      [[0, 1.1],
#       [0, 1.1]],
#
#      [[1.1, 1.1],
#       [1.1, 1.1]]]
#
# a = [[[2, 2],
#       [2, 2]],
#
#      [[3, 3],
#       [3, 3]]]
#
# # print(tf.multiply(x, a))
# weight = tf.get_variable('weight', shape=[10, 10],
#                          initializer=tf.keras.initializers.he_normal())
# print(weight)

#
# labels = [2, 1, 2, 1]
#
#
# def cal_loss(logits, lab_batch):
#     loss = logits - lab_batch
#     return loss
#
#
# class LinerConv3DLayer(tf.keras.layers.Layer):
#     def __init__(self, kernel_size, activate_size, activate_stride):
#         super(LinerConv3DLayer, self).__init__()
#         [self.h, self.w] = kernel_size
#         self.activate_size = activate_size
#         self.activate_stride = activate_stride
#
#         value = [[1, 2],
#                  [3, 4]]
#         init = tf.constant_initializer(value)
#         with tf.name_scope('dont'):
#             self.weight = tf.get_variable('weight', shape=[self.h, self.w], initializer=init)
#
#         # self.b = tf.get_variable('b', shape=[self.h, self.w], initializer=init)
#
#     def __same_padding__(self, inputs):
#         """
#         padding 操作对于矩阵来说是非线性的，因此在矩阵运算后进行 padding 会导致 ValueError: No gradients provided for any variable.
#         但是 tf.pad 会自动忽略这个问题，而np.pad 不行，所以尽量采用 tf 的 API.
#         :param inputs:
#         :return:
#         """
#         h_pad = int(np.ceil((self.h - 1)*self.activate_stride[0] - self.h + self.activate_size[0])/2)
#         w_pad = int(np.ceil((self.w - 1)*self.activate_stride[1] - self.w + self.activate_size[1])/2)
#         padded = tf.pad(inputs, (
#             # (0, 0),  # 样本数，不填充
#             (h_pad, h_pad),  # 图像高度, 上面填充x个，下面填充y个(x,y)
#             (w_pad, w_pad),  # 图像宽度, 左边填充x个，右边填充y个(x,y)
#             # (0, 0)  # 通道数，不填充
#         ), 'constant', constant_values=0)  # 连续一样的值填充
#         return padded
#
#     def __activate_on3x3__(self, inputs):
#         # [h, w] = tf.shape(inputs).numpy()  # 从1开始数
#         # new_map = tf.zeros((self.h, self.w))
#         # print('nnnnnnnnnnn', np.shape(new_map))
#         # a_list = tf.unstack(new_map)
#         # print('aaaaaaaaaaa', np.shape(a_list))
#         #
#         # for i in range(h-2):  # 从0开始数
#         #     for j in range(w-2):
#         #         a_list.append(inputs[i][j] + inputs[i][j+1] + inputs[i][j+2] +\
#         #                         inputs[i+1][j] + inputs[i+1][j+1] + inputs[i+1][j+2] +\
#         #                         inputs[i+2][j] + inputs[i+2][j+1] + inputs[i+2][j+2])
#         # print(a_list)
#         # a_tensor = tf.stack(a_list)
#
#         # for h in range(self.h):  # 在输出的垂直轴上循环
#         #     for w in range(self.w):  # 在输出的水平轴上循环
#         #         # 定位当前的切片位置
#         #         vert_start = h * self.activate_stride[0]  # 竖向，开始的位置
#         #         vert_end = vert_start + self.activate_size[0]  # 竖向，结束的位置
#         #         horiz_start = w * self.activate_stride[1]  # 横向，开始的位置
#         #         horiz_end = horiz_start + self.activate_size[1]  # 横向，结束的位置
#         #         # 定位完毕，开始切割
#         #         a_slice = inputs[vert_start:vert_end, horiz_start:horiz_end]
#
#         out_map = tf.nn.conv2d(inputs, [3, 3], [1, 1], padding='SAME')
#         tf.stop_gradient(out_map)
#
#         return out_map
#
#     def call(self, inputs, training=None, mask=None):
#         inputs = tf.cast(inputs, tf.float32)
#
#         a = tf.multiply(inputs, self.weight)
#
#         print('this is a \n', a)
#         b = self.__same_padding__(a)
#         print('this is b \n', b)
#         # c = self.__activate_on3x3__(b)
#         # print('this is c \n', c)
#
#         return b
#
#
# linearc = LinerConv3DLayer(kernel_size=[2, 2], activate_size=[3, 3], activate_stride=[1, 1])
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#
#
# # 去掉不需要训练的参数
# def get_variable_via_scope(scope_lst):
#     vars = []
#     for scope in scope_lst:
#         sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
#         vars.extend(sc_variable)
#     return vars
#
#
# no_change_scope = ['dont']
# no_change_vars = get_variable_via_scope(no_change_scope)
#
# trainable_vars = linearc.trainable_variables
# print('0000000000000', trainable_vars)
# for v in no_change_vars:
#     trainable_vars.remove(v)
# print('1111111111111111', trainable_vars)
# # 结束
#
# for (img, label) in zip(x, labels):
#     with tf.GradientTape() as tape:
#         y = linearc.call(img)
#         y_ = tf.reduce_sum(y)
#
#         print('this is y, label \n', y_.numpy(), label)
#         loss = cal_loss(y_, label)
#
#     grads = tape.gradient(loss, trainable_vars)
#     optimizer.apply_gradients(zip(grads, trainable_vars))
#
#     print('参数 w \n', linearc.trainable_variables)
#
