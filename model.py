import tensorflow as tf
import sys
import numpy as np


class The3dcnn_lstm_Model(tf.keras.Model):

    def __init__(self, rnn_units, batch_size, num_class):
        super(The3dcnn_lstm_Model, self).__init__()
        self.rnn_units = rnn_units
        # self.batch_size = batch_size
        self.num_class = num_class

        # 3DCNN
        self.conv3d1 = tf.keras.layers.Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last',
                                              name='conv1')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=[1, 2, 2], strides=[1, 2, 2], padding='same',
                                                  data_format='channels_last', name='pool1')

        self.conv3d2 = tf.keras.layers.Conv3D(filters=32, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last',
                                              name='conv2')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_last', name='pool2')

        self.conv3d3 = tf.keras.layers.Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last',
                                              name='conv3')
        self.pooling3 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same',
                                                  data_format='channels_last', name='pool3')

        self.conv3d4 = tf.keras.layers.Conv3D(filters=4, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last',
                                              name='conv4')
        self.pooling4 = tf.keras.layers.MaxPool3D(pool_size=[3, 2, 2], strides=[3, 2, 2], padding='same',
                                                  data_format='channels_last', name='pool4')

        # # RNN
        # self.cell1 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, return_sequences=True, return_state=False)
        # self.cell2 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True, return_state=False)
        # self.cell2 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True, return_state=False)
        #
        # FC
        # self.fc1 = tf.keras.layers.Dense(units=128, use_bias=True, activation=tf.nn.relu,
        #                                  kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
        #                                  bias_initializer=tf.constant_initializer)
        # self.fc2 = tf.keras.layers.Dense(units=num_class, use_bias=True, activation=None,
        #                                  kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
        #                                  bias_initializer=tf.constant_initializer)
        # self.dropout1 = tf.keras.layers.Dropout(0.7)
        # self.dropout2 = tf.keras.layers.Dropout(0.7)

    def call(self, inputs, training=True, **kwargs):
        """
        :param training:
        :param **kwargs:
        :param inputs: [?, 10, 80, 200, 1]
        :return:
        """
        print(type(inputs))
        conv1 = self.conv3d1(inputs)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        pool1 = self.pooling1(conv1)  # (?, 10, 40, 100, 28)
        print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv3d2(pool1)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        pool2 = self.pooling2(conv2)  # (?, 5, 20, 50, 32)
        print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3d3(pool2)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        pool3 = self.pooling3(conv3)  # (?, 3, 10, 25, 16)
        print('pool3: ', pool3.get_shape().as_list())

        conv4 = self.conv3d4(pool3)
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        pool4 = self.pooling4(conv4)  # (?, 1, 5, 13, 4)
        print('pool4: ', pool4.get_shape().as_list())

        x = tf.squeeze(pool4, axis=1)  # (?, 5, 13, 4)
        print('lstm :\n', x.get_shape().as_list())  # [?, 5, 13, 4]
        ##################################################################
        # 版本一 该方法效果不理想, 减小了10个百分点
        # x = tf.transpose(x, [0, 2, 1, 3])  # [?, 25, 10, 16]
        #
        # # print(x.get_shape().as_list())  # [?, 25, 10, 16]
        # # treat `feature_w` as max_timestep in lstm.
        # # vac_len = tf.shape(x)[2] * tf.shape(x)[3]
        # x = tf.reshape(x, [self.batch_size, 25, 160])
        # print('lstm input shape: {}'.format(x.get_shape().as_list()))  # [?, 25, 160]
        #
        # outputs1 = self.cell1(x)  # [?, 25, 1024]
        # outputs2 = self.cell2(outputs1)  # [?, 25, 1024]
        # outputs3 = self.cell2(outputs2)  # [?, 25, 1024]
        #
        # output = outputs3[:, -1, :]  # [?, 1024]
        ####################################################################
        # 版本二
        # h_conv3_features = tf.unstack(x, axis=-1)  # [[?, 5, 13], ....16]
        # channel = x.get_shape().as_list()[-1]
        # rnn_output = []
        # for channel_index in range(channel):
        #     name = "gru_" + str(channel_index)
        #     item_x = tf.transpose(h_conv3_features[channel_index], [0, 2, 1])  # [?, 13, 5] for item in 16
        #     cell1 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, return_sequences=True, return_state=False, name=name)
        #     cell2 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, name=name)
        #
        #     item_out = cell1(inputs=item_x)  # [?, 13, rnn_units]
        #     item_out = cell2(inputs=item_out)  # [?, rnn_units]
        #     cell1 = None
        #     cell2 = None
        #
        #     rnn_output.append(item_out)
        # output = tf.concat(rnn_output, 1)  # [?, self.rnn_units*16]

        # if training:
        #     fc1 = self.fc1(output)
        #     d1 = self.dropout1(fc1)
        #     fc2 = self.fc2(d1)
        #     logits = self.dropout2(fc2)
        # else:
        #     fc1 = self.fc1(output)
        #     logits = self.fc2(fc1)
        #####################################################################
        # 版本三 [?, 5, 13, 4]  14epoch 0.2 0.9
        # h_conv3_features = tf.unstack(x, axis=-1)  # [[?, 5, 13], ....4]
        # channel = x.get_shape().as_list()[-1]
        # rnn_output = []
        # for channel_index in range(channel):
        #     _name_end = str(channel_index)
        #     item_x = tf.transpose(h_conv3_features[channel_index], [0, 2, 1])  # [?, 13, 5] for item in 16
        #     cell1 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, return_sequences=True, return_state=False, name='lstm1'+ _name_end)
        #     cell2 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, name='lstm2'+ _name_end)
        #     dense = tf.keras.layers.Dense(units=1, use_bias=True, activation=None,
        #                                   kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
        #                                   bias_initializer=tf.ones_initializer, name='fc'+ _name_end)
        #
        #     item_out = cell1(inputs=item_x)  # [?, 13, rnn_units]
        #     item_out = cell2(inputs=item_out)  # [?, rnn_units]
        #     fc = dense(item_out)  # [?, 1]
        #
        #     rnn_output.append(fc)
        # rnn_output = tf.squeeze(rnn_output, axis=2)  # [4, ?]
        # logits = tf.transpose(rnn_output)
        # 版本四 # [?, 5, 13, 4]
        x = tf.transpose(x, [0, 2, 1, 3])  # [?, 13, 5, 4]
        batch_size = tf.shape(x)[0]
        time = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, time, 20])
        rnn_output = []
        for index in range(self.num_class):
            _name_end = str(index)
            cell1 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, return_sequences=True, return_state=False, name='lstm1'+ _name_end)
            cell2 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, name='lstm2' + _name_end)
            dense = tf.keras.layers.Dense(units=1, use_bias=True, activation=None,
                                          kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
                                          bias_initializer=tf.ones_initializer, name='fc' + _name_end)

            item_out = cell1(inputs=x)  # [?, 25, rnn_units]
            item_out = cell2(inputs=item_out)  # [?, rnn_units]
            fc = dense(item_out)  # [?, 1]

            rnn_output.append(fc)
        rnn_output = tf.squeeze(rnn_output, axis=2)  # [4, ?]
        logits = tf.transpose(rnn_output)

        print('logits: ', logits)
        return logits
