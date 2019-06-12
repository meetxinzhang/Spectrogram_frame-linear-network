# coding:utf-8
import tensorflow as tf
import model
import math
import input_data
from MyException import MyException
import matplotlib.pyplot as plt
tf.enable_eager_execution()

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
epoch = 2  # 训练的 epoch 数，从1开始计数
display_step = 1

loss_history = []
acc_history = []


logs_path = 'tensor_logs/'
fuckdata = input_data.input_data(file_dir='sounds_data/new_images',
                                 move_stride=move_stride, depth=depth, num_class=num_class)


def my_learning_rate(epoch_index, step):
    if epoch_index != 0:
        return 0.001 * (0.7**(epoch_index-1)) / (1 + step * 0.000001)
        # return 0.001
    else:
        return 0.000001


def cal_loss(logits, lab_batch):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


t3lm = model.The3dcnn_lstm_Model(rnn_units=rnn_units, num_class=num_class)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.8)
# optimizer = tf.keras.optimizers.RMSprop(lr=0.001)

step = 1

try:  # 捕获 input_data 在数据输送结束时的异常
    while True:
        batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
        learning_rate = my_learning_rate(epoch_index, step)
        if epoch_index != 0:
            d_rate = drop_rate
        else:
            d_rate = 0.0

        with tf.GradientTape() as tape:
            logits = t3lm.call(batch_x, drop_rate=d_rate)
            loss = cal_loss(logits, batch_y)

        grads = tape.gradient(loss, t3lm.trainable_variables)
        optimizer.apply_gradients(zip(grads, t3lm.trainable_variables))

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_history.append(loss.numpy())
        acc_history.append(accuracy)

        print('epoch:{}, stpe:{}, loss:{:.3f}, acc:{:.3f}, lr:{:.4f}'.
              format(epoch_index, step, loss, accuracy, learning_rate))
        step += 1

except MyException as e:  # 画图
    fig = plt.figure(figsize=(6, 4))

    plt.plot(loss_history)
    plt.xlabel('Batch [64]')
    plt.ylabel('Loss [entropy]')

    # 设置刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(loc='lower right', fontsize=18)
    plt.show()

