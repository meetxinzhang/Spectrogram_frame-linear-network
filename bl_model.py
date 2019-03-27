import tensorflow as tf
import numpy as np
import bl_data

tf.enable_eager_execution()


class XNN(tf.keras.Model):

    def __init__(self, num_class, rnn_units):
        super(XNN, self).__init__()
        self.num_class = num_class
        self.rnn_units = rnn_units

        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.zeros_initializer())
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.zeros_initializer())
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv3 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.zeros_initializer())
        self.pooling3 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.cell = tf.keras.layers.CuDNNLSTM(units=self.rnn_units)
        self.fc = tf.keras.layers.Dense(units=self.num_class, use_bias=True, activation=None,
                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                        bias_initializer=tf.constant_initializer())

    def call_m(self, input):
        conv1 = self.conv1(input)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        pool1 = self.pooling1(conv1)  # (?, 40, 100, 32)
        print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv2(pool1)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        pool2 = self.pooling2(conv2)  # (?, 20, 50, 64)
        print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3(pool2)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        pool3 = self.pooling3(conv3)  # (?, 10, 25, 16)
        print('pool3: ', pool3.get_shape().as_list())
        return pool3

    def call_rnn(self, input):
        rnn_out = self.cell(input)
        logits = self.fc(rnn_out)
        return logits

    def windows(self, length, window_size):
        start = 0
        while start < length:
            yield start, start + window_size
            start += int(window_size)

    def call(self, input, **kwargs):
        inputs = tf.unstack(input, axis=0)
        batch_features = []
        max = 0
        best_index = 0
        window_size = 200

        for img in inputs:
            shape = np.shape(img)

            for (start, end) in self.windows(shape[1], window_size=window_size):
                if np.shape(img[:, start:end])[1] == window_size:
                    signal = img[:, start:end]
                    cnn_out = self.call_m(signal)  # 调用cnn
                    he = sum(sum(cnn_out))
                    if he >= max:
                        max = he
                        best_index = start

            feature = img[:, best_index:best_index + window_size]
            batch_features.append(np.transpose(feature))

        # 调用rnn
        logits = self.call_rnn(batch_features)
        return logits


# model
depth = 10
height = 80
wigth = 200
chennel = 1

rnn_units = 64
num_class = 4

learning_rate = 0.05
batch_size = 64
epoch = 6
display_step = 1

dd = bl_data.batch_generator(file_dir='yield/images', num_class=num_class)
xnn = XNN(num_class=4, rnn_units=128)


def cal_loss(logits, lab_batch):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


step = 1
while step * batch_size < 99999:
    batch_x, batch_y, epoch_index = dd.next_batch(batch_size=batch_size, epoch=epoch)
    learning_rate = 0.05 / (1 + 0.57 * (epoch_index - 1))
    if epoch_index > epoch:
        t = True
        d_rate = 0.3
    else:
        t = False
        d_rate = 0

    with tf.GradientTape() as tape:
        logits = xnn.call(batch_x)
        loss = cal_loss(logits, batch_y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads = tape.gradient(loss, xnn.variables)
    optimizer.apply_gradients(zip(grads, xnn.variables))

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print('loss:{:.3f}, acc:{:.3f}, lr:{:.4f}'.format(loss, accuracy, learning_rate))
