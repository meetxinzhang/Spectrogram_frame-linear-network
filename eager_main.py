# coding:utf-8
import tensorflow as tf
import model
import math
import time
import input_data
from MyException import MyException
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
tf.enable_eager_execution()

# model
height = 80
width = 6
move_stride = int(width - (width * 0.5))
depth = math.ceil((600 - width) / move_stride) + 1
print('depth:{}, move_stride:{}'.format(depth, move_stride))
rnn_units = 128
drop_rate = 0.3
num_class = 4

batch_size = 92
epoch = 4  # 训练的 epoch 数，从1开始计数
display_step = 1

# data to store
loss_history = []
acc_history = []
test_loss_history = []
test_acc_history = []
best_acc = 0
y_true = []
y_pred = []


def txt_save(data_m, name):
    # logs_path = 'tensor_logs/' + time.strftime(name + "%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
    logs_path = 'tensor_logs/' + name + "_over5" + '.txt'
    file = open(logs_path, 'a')
    for line in data_m:
        for v in line:
            s = str(v) + '\t'
            file.write(s)
        file.write('\n')
    file.close()
    print(name + 'saved')


fuckdata = input_data.input_data(file_dir='sounds_data/new_images',
                                 width=width, move_stride=move_stride, depth=depth, num_class=num_class)


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


the_model = model.Model_X(rnn_units=rnn_units, num_class=num_class)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.8)

step = 1
try:  # 捕获 input_data 在数据输送结束时的异常
    while True:
        batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
        # learning_rate = my_learning_rate(epoch_index, step)
        if epoch_index != 0:
            d_rate = drop_rate
        else:
            d_rate = 0.0

        with tf.GradientTape() as tape:
            logits = the_model.call(batch_x, drop_rate=d_rate)
            loss = cal_loss(logits, batch_y)

        if epoch_index != 0:
            grads = tape.gradient(loss, the_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, the_model.trainable_variables))
        else:
            pass

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        step += 1
        print('epoch:{}, stpe:{}, loss:{:.3f}, acc:{:.3f}'.
              format(epoch_index, step, loss, accuracy))

        # 结果分析
        if epoch_index != 0:
            loss_history.append(loss.numpy())
            acc_history.append(accuracy.numpy())
        else:
            test_loss_history.append(loss.numpy())
            test_acc_history.append(accuracy.numpy())

            # 选择最好的一个批次记录混淆矩阵
            # if best_acc < accuracy.numpy():
            #     y_pred = tf.math.argmax(logits, axis=1).numpy()
            #     y_true = tf.math.argmax(batch_y, axis=1).numpy()
            #     best_acc = accuracy.numpy()

            for l in tf.math.argmax(logits, axis=1).numpy():
                y_pred.append(l)
            for y in tf.math.argmax(batch_y, axis=1).numpy():
                y_true.append(y)


except MyException as e:  # 画图
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors1 = 'C0'
    colors2 = 'C1'

    axs[0, 0].plot(acc_history, label='train', color=colors1)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('accuracy')

    axs[0, 1].plot(loss_history, label='train', color=colors1)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].set_xlabel('step')
    axs[0, 1].set_ylabel('loss')

    axs[1, 0].plot(test_acc_history, label='test', color=colors1)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('accuracy')

    axs[1, 1].plot(test_loss_history, label='test', color=colors1)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].set_ylabel('loss')

    plt.show()

    # 保存日志文件
    data_m = [loss_history, acc_history, test_loss_history, test_acc_history]
    txt_save(data_m, name='lines')
    txt_save([y_pred, y_true], name='y_')
