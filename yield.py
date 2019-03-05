# coding=utf-8

import os
import re
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from random import shuffle
import librosa

path = "sounds/Phasianus+colchicus/Phasianus colchicus/"

# learning_rate = 0.00001
# training_iters = 300000 #steps
# batch_size = 64

height = 20  # mfcc features
width = 80  # (max) length of utterance
classes = 10  # digits

n_input = 20
n_steps = 80
n_hidden = 128
n_classes = 10

learning_rate = 0.02
training_iters = 1000
batch_size = 10
display_step = 5
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def mfcc_batch_generator(batch_size=10):
    # maybe_download(source, DATA_DIR)
    batch_features = []
    labels = []
    files = os.listdir(path)
    while True:
        # print("loaded batch of %d files" % len(files))
        shuffle(files)
        for file in files:
            if not file.endswith(".mp3"):
                continue
            wave, sr = librosa.load(path+file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            label = dense_to_one_hot(int(file[0]), 2)
            labels.append(label)
            print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80-len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc).T)
            if len(batch_features) >= batch_size:
                yield np.array(batch_features), np.array(labels)
                batch_features = []  # Reset for next batch
                labels = []


def dense_to_one_hot(labels_dense, num_classes=10):
    return np.eye(num_classes)[labels_dense]


def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch = mfcc_batch_generator(batch_size)
        batch_x, batch_y = next(batch)
        # print(batch_x.shape)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y : batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
                "{:.6f}".format(loss) + ", Training Accuracy = " + \
                "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

