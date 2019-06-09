# coding=utf-8
import tensorflow as tf
import model
import time
import math
import input_data

sess = tf.InteractiveSession()

# model
move_stride = int(200-(200*0.8))
depth = math.ceil((600-200)/move_stride)+1
print('depth:{}, move_stride:{}'.format(depth, move_stride))
height = 80
wigth = 200
chennel = 1
rnn_units = 64
drop_rate = 0.3
num_class = 4

batch_size = 64
epoch = 3  # 训练的 epoch 数，从1开始计数
display_step = 1


def my_learning_rate(epoch_index, step):
    if epoch_index != 0:
        return 0.001 * (0.7**(epoch_index-1)) / (1 + step * 0.000001)
        # return 0.001
    else:
        return 0.000001



"""
activate tf
tensorboard --logdir=tensor_logs
"""
logs_path = 'tensor_logs/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
fuckdata = input_data.input_data(file_dir='sounds_data/new_images',
                                 move_stride=move_stride, depth=depth, num_class=num_class)

x_ph = tf.placeholder("float", [None, chennel, depth, height, wigth])
y_ph = tf.placeholder("float", [None, num_class])
drop_rate_ph = tf.placeholder(tf.float32)
learning_rate_ph = tf.placeholder(tf.float32)

# 定义global_step
# global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率
# learing_rate = tf.train.exponential_decay(0.1, global_step, 64, 0.7, staircase=False)

##############################################
t3lm = model.The3dcnn_lstm_Model(rnn_units=rnn_units, num_class=num_class)
logits = t3lm.call(x_ph, drop_rate=drop_rate_ph)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_ph))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph, momentum=0.8).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(cost)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_ph, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# 创建一个summary来监测成本
tf.summary.scalar("loss", cost)
# 创建一个summary来监测精确度
tf.summary.scalar("accuracy", accuracy)
# 创建一个summary来监测学习率
tf.summary.scalar("learning_rate", learning_rate_ph)
# 将所有summary合并为一个操作op
merged_summary_op = tf.summary.merge_all()

sess.run(init)

# 定义Tensorboard的事件文件路径
summary_train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
summary_test_writer = tf.summary.FileWriter(logs_path + '/test')

step = 1
first = True
while True:
    batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
    lr = my_learning_rate(epoch_index, step)

    if epoch_index != 0:
        d_rate = drop_rate
        _, summary = sess.run([optimizer, merged_summary_op],
                              feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate, learning_rate_ph: lr})
        summary_train_writer.add_summary(summary, step)  # 在每次迭代中将数据写入事件文件
    else:
        if first:
            step = 0; first = False  # 测试时将step归零
        d_rate = 0
        _, summary = sess.run([optimizer, merged_summary_op],
                              feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate, learning_rate_ph: lr})
        summary_test_writer.add_summary(summary, step)  # 在每次迭代中将数据写入事件文件

    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate})
        loss = sess.run(cost, feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate})
        # lr = sess.run(learing_rate)
        print('epoch={},step={},[loss={:.3f},acc={:.3f}],lr={:.6f}'.format(epoch_index, step, loss, acc, lr))
    step += 1


# 对比实验 超参数日志
pass
# id=1
# 3d cnn + fc
# over=0.8 acc=0.81
# batch_size = 32
# epoch = 3
# lr = 0.0005
# AdamOptimizer
pass
# id=2
# 3d cnn + lstm + fc
# over=0.8 acc = 0.89
# batch_size = 64
# epoch = 3
# lr = 0.005
# AdamOptimizer
pass
# id=2019-04-24-15-20-54
# 3d cnn + lstm + fc
# over=0.8 acc = 0.92
# momentum=0.8
# batch_size = 92
# epoch = 3
# lr = 0.005 * (0.7**(epoch_index-1)) / (1 + step * 0.005)
# RMSPropOptimizer
pass
# id=2019-04-24-16-02-05
# 3d cnn + lstm分通道 + fc
# acc = 0.85
# 其他同上
pass
# id=2019-04-24-16-08-04
# 3d cnn + 一个lstm + fc
# acc = 0.88
# 其他同上
pass
