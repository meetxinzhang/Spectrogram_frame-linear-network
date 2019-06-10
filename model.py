import tensorflow as tf
from PIL import Image
import scipy.misc
import os


class The3dcnn_lstm_Model(tf.keras.Model):

    def __init__(self, rnn_units, num_class):
        super(The3dcnn_lstm_Model, self).__init__()
        self.rnn_units = rnn_units
        self.num_class = num_class

        # 3DCNN
        self.conv3d1 = tf.keras.layers.Conv3D(filters=8, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.leaky_relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_first')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_first')

        self.conv3d2 = tf.keras.layers.Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.leaky_relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_first')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_first')

        self.conv3d3 = tf.keras.layers.Conv3D(filters=8, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.leaky_relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_first')
        self.pooling3 = tf.keras.layers.MaxPool3D(pool_size=[3, 2, 2], strides=[3, 2, 2], padding='same',
                                                  data_format='channels_first')

        self.cell1 = tf.keras.layers.CuDNNGRU(units=80, return_sequences=True)
        self.cell2 = tf.keras.layers.CuDNNGRU(units=self.num_class)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
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
        :param **kwargs:
        :param **kwargs:
        :param input: [?, 1, 11, 80, 200]
        :return:
        """
        # print('inputs: ', np.shape(inputs))
        is_training = tf.equal(drop_rate, 0.3)

        conv1 = self.conv3d1(inputs)
        conv1 = self.bn1(conv1, training=is_training)
        pool1 = self.pooling1(conv1)  # (?, 8, 6, 40, 100)
        # print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv3d2(pool1)
        conv2 = self.bn2(conv2, training=is_training)
        pool2 = self.pooling2(conv2)  # (?, 16, 3, 20, 50)
        # print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3d3(pool2)
        conv3 = self.bn3(conv3, training=is_training)
        pool3 = self.pooling3(conv3)  # (?, 8, 1, 10, 25)
        # print('pool3: ', pool3.get_shape().as_list())

        x_rnn = tf.squeeze(pool3, axis=2)  # (?, 8, 10, 25)
        if not is_training:
            self.draw_hid_features(inputs, x_rnn)
        ##################################################################
        # data_format='channels_last'
        # x_rnn = tf.transpose(x_rnn, [0, 2, 1, 3])  # [?, 25, 10, 8]
        # x_rnns = tf.unstack(x_rnn, axis=-1)  # 展开通道维度  [?, 25, 10] * 8
        # x_rnn = tf.concat(x_rnns, axis=-1)  # 合并列维度  [?, 25, 80]
        # data_format='channels_first'
        x_rnns = tf.unstack(x_rnn, axis=1)  # 展开通道维度  [8, ?, 10, 25]
        x_rnn = tf.concat(x_rnns, axis=1)  # 合并列维度  [?, 80, 25]
        x_rnn = tf.transpose(x_rnn, [0, 2, 1])  # [?, 25, 80]

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
        :param inputs: [?, 1, 11, 80, 200]
        :param batch: [?, 8, 10, 25]
        :return: 画图
        """
        import numpy
        inputs = numpy.squeeze(inputs)  # [?, 11, 80, 200]
        batch = batch.numpy()

        index_batch = 0
        for sample in batch:
            # [8, 10, 25]
            index_chennel = 0

            yuan_tus = inputs[index_batch, :, :, :]
            y1 = yuan_tus[0]
            y2 = yuan_tus[5]
            y3 = yuan_tus[10]
            yuan_tu = numpy.hstack([y1, y2, y3])
            save_dir = 'hid_pic' + '/' + str(index_batch) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Image.fromarray(yuan_tu).convert('RGB').save(save_dir + 'yuan_tu.jpg')

            for feature in sample:
                # [10, 25]u
                save_path = 'hid_pic' + '/' + str(index_batch) + '/' + str(index_chennel) + '.jpg'
                scipy.misc.imsave(save_path, feature)
                # Image.fromarray(feature).convert('RGB').save(save_path)

                index_chennel += 1
            index_batch += 1




