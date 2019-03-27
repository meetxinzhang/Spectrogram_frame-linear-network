# coding=utf-8
import tensorflow as tf
import model
import fuck
import time

sess = tf.InteractiveSession()

# model
depth = 5
height = 80
wigth = 200
chennel = 1
rnn_units = 200
num_class = 4

training_iters = 99999
batch_size = 64
epoch = 4
display_step = 1
drop_rate = 0.2


def my_learning_rate(epoch_index, step):
    return 0.001 * (0.5**(epoch_index-1)) / (1 + step * 0.01)



"""
activate tf
tensorboard --logdir=tensor_logs
"""
logs_path = 'tensor_logs/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
path = 'sounds/'
fuckdata = fuck.input_data(train_file_dir=path, depth=depth, height=height, width=wigth, num_class=num_class)

x_ph = tf.placeholder("float", [None, depth, height, wigth, chennel])
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
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learing_rate, momentum=0.9).minimize(cost, global_step)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph, momentum=0.9).minimize(cost)
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
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
    lr = my_learning_rate(epoch_index, step)
    if epoch_index < epoch:
        d_rate = drop_rate
    else:
        d_rate = 0

    _, summary = sess.run([optimizer, merged_summary_op],
                          feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate, learning_rate_ph: lr})
    # 在每次迭代中将数据写入事件文件
    summary_writer.add_summary(summary, step)
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate})
        loss = sess.run(cost, feed_dict={x_ph: batch_x, y_ph: batch_y, drop_rate_ph: d_rate})
        # lr = sess.run(learing_rate)
        print('epoch={},item={}, [loss={:.3f},acc={:.3f}], lr={:.6f}'.format(epoch_index, step*batch_size, loss, acc, lr))
    step += 1
print("Optimization Finished!")
