# coding:utf-8
import tensorflow as tf
import model
import input_data
tf.enable_eager_execution()

# model
depth = 5
height = 80
wigth = 200
chennel = 1

rnn_units = 200
num_class = 3

batch_size = 64
epoch = 4
display_step = 1

logs_path = 'tensor_logs/'
fuckdata = input_data.input_data(file_dir='sounds_data/new_images', depth=depth, height=height, width=wigth, num_class=num_class)


def my_learning_rate(epoch_index, step):
    if epoch_index != 0:
        return 0.005 * (0.5**(epoch_index-1)) / (1 + step * 0.01)
    else:
        return 0.000001


def cal_loss(logits, lab_batch):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


t3lm = model.The3dcnn_lstm_Model(rnn_units=rnn_units, num_class=num_class)

step = 1
while step * batch_size < 9999:
    batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
    learning_rate = my_learning_rate(epoch_index, step)
    if epoch_index != 0:
        d_rate = 0.2
    else:
        d_rate = 0

    with tf.GradientTape() as tape:
        logits = t3lm.call(batch_x, drop_rate=d_rate)
        loss = cal_loss(logits, batch_y)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    grads = tape.gradient(loss, t3lm.variables)
    optimizer.apply_gradients(zip(grads, t3lm.variables))

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print('loss:{:.3f}, acc:{:.3f}, lr:{:.4f}'.format(loss, accuracy, learning_rate))
