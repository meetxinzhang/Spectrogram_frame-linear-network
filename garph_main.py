# coding=utf-8
import tensorflow as tf
import model
import fuck

sess = tf.InteractiveSession()

# model
depth = 10
height = 80
wigth = 200
chennel = 1

rnn_units = 64
num_class = 4

# learning_rate = 0.05
training_iters = 1582 * 50
batch_size = 64
epoch = 15
display_step = 1

"""
activate tf
tensorboard --logdir=tensor_logs
"""
logs_path = 'tensor_logs/'
path = 'sounds/'
fuckdata = fuck.input_data(train_file_dir=path, depth=depth, height=height, width=wigth, num_class=num_class)

x = tf.placeholder("float", [None, depth, height, wigth, chennel])
y = tf.placeholder("float", [None, num_class])
drop_rate = tf.placeholder(tf.float32)
# lr = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

# 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率
learing_rate = tf.train.exponential_decay(0.05, global_step, 1, 0.96, staircase=False)

##############################################
t3lm = model.The3dcnn_lstm_Model(rnn_units=rnn_units, num_class=num_class)
logits = t3lm.call(x, training=training, dropout=drop_rate)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learing_rate, momentum=0.9).minimize(cost, global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost, global_step)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# 创建一个summary来监测成本
tf.summary.scalar("loss", cost)
# 创建一个summary来监测精确度
tf.summary.scalar("accuracy", accuracy)
# 创建一个summary来监测学习率
tf.summary.scalar("learning_rate", learing_rate)
# 将所有summary合并为一个操作op
merged_summary_op = tf.summary.merge_all()

sess.run(init)

# 定义Tensorboard的事件文件路径
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
    # learning_rate = 0.05/(1+0.05*(epoch_index-1))
    if epoch_index < epoch:
        t = True
        d_rate = 0
    else:
        t = False
        d_rate = 0
    # batch_x = batch_x.reshape((batch_size, height, wigth))
    _, summary = sess.run([optimizer, merged_summary_op],
                          feed_dict={x: batch_x, y: batch_y, training: t, drop_rate: d_rate})
    # 在每次迭代中将数据写入事件文件
    summary_writer.add_summary(summary, step)
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, training: t, drop_rate: d_rate})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, training: t, drop_rate: d_rate})
        lr = sess.run(learing_rate)
        print("Iter " + str(step * batch_size) + ", batch Loss = " +
              "{:.4f}".format(loss) + ", Training Accuracy = " +
              "{:.4f}".format(acc) + ", lr=" +
              "{:.3f}".format(lr))
    step += 1
print("Optimization Finished!")
