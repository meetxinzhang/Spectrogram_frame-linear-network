# coding: utf-8
# ---
# @File: eager_main.py
# @description: 主函数，使用 tensorflow eager 模式
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月18, 2019
# ---

import tensorflow as tf
import model
import math
import time
import input_data
from MyException import MyException
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
# tf.enable_eager_execution()

# model
height = 80
width = 6
move_stride = int(width - (width * 0.25))
depth = math.ceil((600 - width) / move_stride) + 1
print('depth:{}, move_stride:{}'.format(depth, move_stride))
rnn_units = 64
drop_rate = 0.3
num_class = 4

batch_size = 8
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
    """
    以 txt 保存实验日志，tensorboard 画的图实在太丑了，并且不支持 eager 模式，只好自己写一个；
    该 txt 文件，在draw_cm.py, draw_many_line.py, draw_single_line.py 中均支持，txt 文件说明如下：
    ————————————————————
    训练损失------line
    训练准确率----line
    测试损失------line
    测试准确率----line
    ————————————————————
    :param data_m: 数据list
    :param name: 文件名
    """
    logs_path = 'tensor_logs/' + time.strftime(name + "%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
    # logs_path = 'tensor_logs/' + name + "_over5" + '.txt'
    file = open(logs_path, 'a')
    for line in data_m:
        for v in line:
            s = str(v) + '\t'
            file.write(s)
        file.write('\n')
    file.close()
    print(name + 'saved')


# 初始化 input_data 类的对象
fuckdata = input_data.input_data(file_dir='E:/数据集/sounds_data/new_images',
                                 width=width, move_stride=move_stride, depth=depth, num_class=num_class)


def my_learning_rate(epoch_index, step):
    if epoch_index != 0:
        return 0.001 * (0.7**(epoch_index-1)) / (1 + step * 0.000001)
        # return 0.001
    else:
        return 0.000001


def cal_loss(logits, lab_batch):
    """
    计算损失
    :param logits: 模型输出
    :param lab_batch: 标签 batch
    :return: loss，tensor 类型
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


# 初始化模型和优化器
the_model = model.Model_X(rnn_units=rnn_units, num_class=num_class)
optimizer = tf.optimizers.RMSprop(learning_rate=0.0008, momentum=0.7)
# 获取模型中可训练的参数
trainable_vas = the_model.trainable_variables
print('trainable_vas', len(trainable_vas))

step = 1  # 训练step，一个 step 处理一个 batch 的数据
try:
    while True:  # 从训练到测试的节奏由 fuckdata.next_batch 控制，因此写个死循环就行
        batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
        # learning_rate = my_learning_rate(epoch_index, step)
        if epoch_index != 0:
            d_rate = drop_rate  # 判定训练
        else:
            d_rate = 0.0  # 判定测试

        # 记录梯度
        with tf.GradientTape() as tape:
            logits = the_model.call(batch_x, drop_rate=d_rate)
            loss = cal_loss(logits, batch_y)

        # 如果为训练阶段，则应用梯度下降，让模型学习；测试阶段什么都不做
        if epoch_index != 0:
            grads = tape.gradient(loss, trainable_vas)
            optimizer.apply_gradients(zip(grads, trainable_vas))
        else:
            pass

        # 计算准确率
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        step += 1  # step 自增
        print('epoch:{}, stpe:{}, loss:{:.3f}, acc:{:.3f}'.
              format(epoch_index, step, loss, accuracy))

        # 记录每个 step 的实验日志
        if epoch_index != 0:
            loss_history.append(loss.numpy())
            acc_history.append(accuracy.numpy())
        else:
            test_loss_history.append(loss.numpy())
            test_acc_history.append(accuracy.numpy())

            # 测试阶段，选择最好的一个批次，记录预测值和标签值，用于混淆矩阵分析
            # if best_acc < accuracy.numpy():
            #     y_pred = tf.math.argmax(logits, axis=1).numpy()
            #     y_true = tf.math.argmax(batch_y, axis=1).numpy()
            #     best_acc = accuracy.numpy()
            # 测试阶段，记录全部批次的记录预测值和标签值，用于混淆矩阵分析
            for l in tf.math.argmax(logits, axis=1).numpy():
                y_pred.append(l)
            for y in tf.math.argmax(batch_y, axis=1).numpy():
                y_true.append(y)


except MyException as e:
    # 捕获 input_data 在数据输送结束时的异常，开始画图
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
