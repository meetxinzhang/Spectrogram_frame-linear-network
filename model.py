import tensorflow as tf


class The3dcnn_lstm_Model(tf.keras.Model):

    def __init__(self, rnn_units, batch_size, num_class):
        super(The3dcnn_lstm_Model, self).__init__()
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.num_class = num_class

        # 3DCNN
        self.conv3d1 = tf.keras.layers.Conv3D(
            filters=32, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
        self.conv3d2 = tf.keras.layers.Conv3D(
            filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
        self.conv3d3 = tf.keras.layers.Conv3D(
            filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling3 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 3], strides=[2, 2, 3], padding='same')
        # Reshape
        self.flatten = tf.keras.layers.Reshape(target_shape=[-1, 10, 25, 16])

        # LSTM
        if tf.test.is_gpu_available:
            self.gru1 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True, return_state=True)
            self.gru2 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True, return_state=True)
            self.gru3 = tf.keras.layers.CuDNNGRU(units=self.rnn_units, return_sequences=True, return_state=True)
        else:
            print('gpu 不可用')
            assert False
        # FC
        self.fc = tf.keras.layers.Dense(units=num_class, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.7)

    def call(self, inputs, training=False, **kwargs):
        """
        :param **kwargs:
        :param **kwargs:
        :param input: (?, 80, 200, 10, 1)
        :return:
        """
        conv1 = self.conv3d1(inputs)
        pool1 = self.pooling1(conv1)  # (?, 40, 100, 5, 32)
        # print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv3d2(pool1)
        pool2 = self.pooling2(conv2)  # (?, 20, 50, 3, 64)
        # print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3d3(pool2)
        pool3 = self.pooling3(conv3)  # (?, 10, 25, 1, 16)
        # print('pool3: ', pool3.get_shape().as_list())

        x = tf.squeeze(pool3)  # (?, 10, 25, 16)
        # print('lstm :\n', x.get_shape().as_list())  # [?, 10, 25, 16]
        x = tf.transpose(x, [0, 2, 1, 3])

        # print(x.get_shape().as_list())  # [?, 25, 10, 16]
        # treat `feature_w` as max_timestep in lstm.
        x = tf.reshape(x, [self.batch_size, -1, self.rnn_units])
        # print('lstm input shape: {}'.format(x.get_shape().as_list()))  # [?, 25, 160]

        outputs1, _ = self.gru1(x)  # [?, 25, 160]
        outputs2, _ = self.gru2(outputs1)  # [?, 25, 160]
        outputs3, h_state = self.gru3(outputs2)  # [?, 25, 160]

        output = outputs3[:, -1, :]  # [?, 160]

        if training:
            d = self.dropout(output)
            logits = self.fc(d)
        else:
            logits = self.fc(output)

        print('logits: ', logits)
        return logits

