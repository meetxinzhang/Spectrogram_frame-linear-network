import tensorflow as tf
import base_line_data


class CNN(tf.keras.Model):

    def __init__(self, batch_size, num_class):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.num_class = num_class

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv3 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling3 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv4 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling4 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv5 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=tf.nn.relu, padding='same')
        self.pooling5 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        # FC
        self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dropout1 = tf.keras.layers.Dropout(0.7)

        self.fc2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(0.7)

        self.fc3 = tf.keras.layers.Dense(units=num_class, activation=tf.nn.relu)

    def call(self, inputs, **kwargs):
        """
        :param **kwargs:
        :param **kwargs:
        :param input: [?,512, 900, 1]
        :return:
        """
        conv1 = self.conv1(inputs)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        pool1 = self.pooling1(conv1)  # (?, 256, 450, 32)
        print('pool1: ', pool1.get_shape().as_list())

        conv2 = self.conv2(pool1)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        pool2 = self.pooling2(conv2)  # (?, 128, 225, 64)
        print('pool2: ', pool2.get_shape().as_list())

        conv3 = self.conv3(pool2)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        pool3 = self.pooling3(conv3)  # (?, 64, 113, 32)
        print('pool3: ', pool3.get_shape().as_list())

        conv4 = self.conv4(pool3)
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        pool4 = self.pooling4(conv4)  # (?, 32, 56, 16)
        print('pool4: ', pool4.get_shape().as_list())

        conv5 = self.conv5(pool4)
        conv5 = tf.layers.batch_normalization(conv5, training=True)
        pool5 = self.pooling5(conv5)  # (?, 16, 28, 8)
        print('pool4: ', pool5.get_shape().as_list())

        w = pool5.get_shape().as_list()[1]
        h = pool5.get_shape().as_list()[2]
        c = pool5.get_shape().as_list()[3]
        pool5 = tf.reshape(pool5, [self.batch_size, w*h*c])
        print('pool4: ', pool5.get_shape().as_list())

        fc1 = self.fc1(pool5)
        d1 = self.dropout1(fc1)

        fc2 = self.fc2(d1)
        d2 = self.dropout2(fc2)

        fc3 = self.fc3(d2)

        return fc3


# coding=utf-8
import tensorflow as tf
import model
from tensorflow.contrib import rnn
import numpy as np
import fuck

sess = tf.InteractiveSession()

# learning_rate = 0.00001
# training_iters = 300000 #steps
# batch_size = 64

# model
height = 400
wigth = 800
chennel = 1

num_class = 8

learning_rate = 0.001
training_iters = 2500
batch_size = 50
display_step = 1

path = 'sounds/'
data = base_line_data.base_line_data(train_file_dir=path, height=height, width=wigth, num_class=num_class)

# [5, 80, 200, 1]

x = tf.placeholder("float", [None, height, wigth, chennel])
y = tf.placeholder("float", [None, num_class])

##############################################
cnn = CNN(batch_size=batch_size, num_class=num_class)
pred = cnn.call(x, training=True)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

sess.run(init)

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y = data.next_batch(batch_size=batch_size)
    # batch_x = batch_x.reshape((batch_size, height, wigth))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss = " +
            "{:.6f}".format(loss) + ", Training Accuracy = " +
            "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")




