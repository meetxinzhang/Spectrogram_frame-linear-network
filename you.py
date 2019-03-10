# coding=utf-8
import tensorflow as tf
import model
import numpy as np
import fuck

sess = tf.InteractiveSession()

# model
depth = 10
height = 80
wigth = 200
chennel = 1

rnn_units = 128
num_class = 4

learning_rate = 0.05
training_iters = 1582 * 50
batch_size = 64
epoch = 1
display_step = 1

# tensorboard --logdir=tensor_logs
logs_path = 'tensor_logs/'
path = ['sounds/', 'sounds_test/']
fuckdata = fuck.input_data(train_file_dir=path[0], test_file_dir=path[1], depth=depth, height=height, width=wigth, num_class=num_class)

# [5, 80, 200, 1]

x = tf.placeholder("float", [None, depth, height, wigth, chennel])
y = tf.placeholder("float", [None, num_class])
lr = tf.placeholder(tf.float32)

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
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# 创建一个summary来监测成本
tf.summary.scalar("loss", cost)
# 创建一个summary来监测精确度
tf.summary.scalar("accuracy", accuracy)
# 将所有summary合并为一个操作op
merged_summary_op = tf.summary.merge_all()

sess.run(init)

# 定义Tensorboard的事件文件路径
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
    learning_rate = 0.05 * (0.57 ** epoch_index)

    # batch_x = batch_x.reshape((batch_size, height, wigth))
    _, summary = sess.run([optimizer, merged_summary_op], feed_dict={x: batch_x, y: batch_y, lr: learning_rate})
    # 在每次迭代中将数据写入事件文件
    summary_writer.add_summary(summary, step)
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step * batch_size) + ", batch Loss = " +
              "{:.6f}".format(loss) + ", Training Accuracy = " +
              "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")
