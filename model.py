import tensorflow as tf


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
                                              data_format='channels_last')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=[3, 2, 2], strides=[3, 2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv3d2 = tf.keras.layers.Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.leaky_relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=[3, 2, 2], strides=[3, 2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv3d3 = tf.keras.layers.Conv3D(filters=4, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
                                              activation=tf.nn.leaky_relu, padding='same',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last')
        self.pooling3 = tf.keras.layers.MaxPool3D(pool_size=[3, 2, 2], strides=[3, 2, 2], padding='same',
                                                  data_format='channels_last')

        # self.conv3d4 = tf.keras.layers.Conv3D(filters=4, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True,
        #                                       activation=tf.nn.leaky_relu, padding='same',
        #                                       kernel_initializer=tf.keras.initializers.he_normal(),
        #                                       bias_initializer=tf.zeros_initializer(),
        #                                       data_format='channels_last')
        # self.pooling4 = tf.keras.layers.MaxPool3D(pool_size=[3, 1, 1], strides=[3, 1, 1], padding='same',
        #                                           data_format='channels_last')

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
        :param input: [?, 21, 80, 200, 1]
        :return:
        """
        is_training = tf.equal(drop_rate, 0.3)

        conv1 = self.conv3d1(inputs)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        pool1 = self.pooling1(conv1)  # (?, 7, 40, 100, 8)
        # print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv3d2(pool1)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        pool2 = self.pooling2(conv2)  # (?, 3, 20, 50, 16)
        # print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3d3(pool2)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        pool3 = self.pooling3(conv3)  # (?, 1, 10, 25, 4)
        # print('pool3: ', pool3.get_shape().as_list())

        # conv4 = self.conv3d4(pool3)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        # pool4 = self.pooling4(conv4)  # (?, 1, 10, 25, 4)

        x_rnn = tf.squeeze(pool3, axis=1)  # (?, 10, 25, 4)
        ##################################################################
        x_rnn = tf.transpose(x_rnn, [0, 2, 1, 3])  # [?, 25, 10, 4]
        x_rnns = tf.unstack(x_rnn, axis=-1)  # 展开通道维度  [?, 25, 10] * 4
        x_rnn = tf.concat(x_rnns, axis=-1)  # 合并列维度  [?, 25, 40]

        rnn_output = []
        for i in range(self.num_class):
            name = "ltsm_" + str(i)
            cell = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, name=name)
            fc = tf.keras.layers.Dense(units=1, use_bias=True, activation=None,
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       bias_initializer=tf.constant_initializer())
            drop = tf.keras.layers.Dropout(rate=drop_rate)

            item_out = cell(inputs=x_rnn)  # [?, 64]
            fc_out = drop(item_out)
            fc_out2 = fc(fc_out)  # [?, 1]
            cell = None
            drop = None
            fc = None

            rnn_output.append(fc_out2)  # [4, ?, 1]
        rnn_output = tf.squeeze(rnn_output)  # [4, ?]
        logits = tf.transpose(rnn_output)  # [?, 4]
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
        # cell = tf.keras.layers.CuDNNLSTM(units=self.rnn_units, return_sequences=True)
        # cell2 = tf.keras.layers.CuDNNLSTM(units=self.rnn_units)
        # drop = tf.keras.layers.Dropout(rate=drop_rate)
        #
        # cell_out1 = cell(x_rnn)
        # cell_out2 = cell2(cell_out1)
        # cell_out = drop(cell_out2)
        # logits = self.fc2(cell_out)

        return logits
