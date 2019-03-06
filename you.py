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
depth = 10
height = 80
wigth = 200
chennel = 1

rnn_units = 32
num_class = 8

learning_rate = 0.01
training_iters = 2500
batch_size = 50
display_step = 1

path = 'sounds/'
fuckdata = fuck.input_data(train_file_dir=path, depth=depth, height=height, width=wigth, num_class=num_class)

# [5, 80, 200, 1]

x = tf.placeholder("float", [None, depth, height, wigth, chennel])
y = tf.placeholder("float", [None, num_class])

##############################################
t3lm = model.The3dcnn_lstm_Model(rnn_units=rnn_units, batch_size=batch_size, num_class=num_class)
pred = t3lm.call(x, training=True)

##############################################
# x = tf.placeholder("float", [None, height, wigth])
# y = tf.placeholder("float", [None, n_classes])
#
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
# }
#
# biases = {
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
#
# def RNN(x, weights, biases):
#     x = tf.unstack(x, n_steps, 1)
#     lstm_cell = rnn.BasicLSTMCell(rnn_units, forget_bias=1.0)
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
#
#
# pred = RNN(x, weights, biases)
#############################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

sess.run(init)

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y = fuckdata.next_batch(batch_size=batch_size)
    # batch_x = batch_x.reshape((batch_size, height, wigth))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
            "{:.6f}".format(loss) + ", Training Accuracy = " + \
            "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")













# import tensorflow as tf
# from tensorflow.contrib import rnn
# import numpy as np
# import fuck
#
#
# path = 'sounds/'
# fuckdata = fuck.input_data(path)
#
# # learning_rate = 0.00001
# # training_iters = 300000 #steps
# # batch_size = 64
#
# height = 80  # mfcc features
# width = 100  # (max) length of utterance
# classes = 8  # digits
#
# n_input = 80
# n_steps = 100
# n_hidden = 100
# n_classes = 8
#
# learning_rate = 0.001
# training_iters = 100000
# batch_size = 10
# display_step = 1
#
# x = tf.placeholder("float", [None, n_steps, n_input])
# y = tf.placeholder("float", [None, n_classes])
#
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
# }
#
# biases = {
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
#
# def RNN(x, weights, biases):
#     x = tf.unstack(x, n_steps, 1)
#     lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
#
#
# pred = RNN(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     while step * batch_size < training_iters:
#         batch_x, batch_y = fuckdata.next_batch(batch_size=batch_size, num_class=classes)
#         batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#         if step % display_step == 0:
#             acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#             loss = sess.run(cost, feed_dict={x: batch_x, y : batch_y})
#             print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
#                 "{:.6f}".format(loss) + ", Training Accuracy = " + \
#                 "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")
