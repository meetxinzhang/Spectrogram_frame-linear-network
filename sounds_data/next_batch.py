# coding: utf-8
# ---
# @File: next_batch.py
# @description: 数据发动机，从本地文件夹生成 batch，自驱动，自停止，训练完自动切换测试，
#   在 main_eager.py 中被调用，只需调用 next_batch 方法
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月18, 2019
# ---

import os
import numpy as np
import build_3d_input
from utils.MyException import MyException
from class_names import class_names


class BatchLoader(object):
    # TODO parallel
    def __init__(self, file_dir, width=200, move_stride=100, depth=10, num_class=4):
        self.file_dir = file_dir
        self.training = True  # 指示当前状态是训练还是测试
        self.epoch_index = 1  # epoch 次数指针，训练从1开始计数，训练数据输送完会指0，开始输送测试数据，next_batch方法会给调用者返回这个值
        self.file_point = 0  # epoch 内的文件指针，每一个新的 epoch 重新归 0

        self.depth = depth
        self.width = width
        self.move_stride = move_stride
        self.num_class = num_class  # 数据集的类别数

        self.train_fnames, self.train_labs, self.test_fnames, self.test_labs\
            = self.get_filenames(self.file_dir)

    def get_filenames(self, file_dir):
        """
        遍历全部本地文件名，赋标签；随机打乱顺序；按0.3的比例划分出测试集
        :param file_dir: 文件根目录
        :return: 四个list
        """
        filenames = []
        labels = []

        for train_class in os.listdir(file_dir):
            for pic in os.listdir(file_dir + '/' + train_class):
                if os.path.isfile(file_dir + '/' + train_class + '/' + pic):
                    filenames.append(file_dir + '/' + train_class + '/' + pic)
                    label = class_names.index(train_class)
                    labels.append(int(label))

        temp = np.array([filenames, labels])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        # 随机打乱顺序
        np.random.shuffle(temp)
        filenames_list = list(temp[:, 0])
        lab_list = list(temp[:, 1])

        n_total = len(filenames_list)
        n_test = int(n_total*0.3)

        test_fnames = filenames_list[0:n_test]
        test_labs = lab_list[0:n_test]
        train_fnames = filenames_list[n_test+1:-1]
        train_labs = lab_list[n_test+1:-1]

        # labels = [int(i) for i in labels]
        print("训练数据 ：", n_total-n_test)
        print("测试数据 ：", n_test)

        return train_fnames, train_labs, test_fnames, test_labs

    def next_batch(self, batch_size, epoch=1):
        """
        获取下一批次（训练或测试）数据
        :param batch_size: 批次大小
        :param epoch: 需要训练的 epoch数，即训练数据集重复训练的遍数
        :return: 数据 numpy数组，标签 numpy数组，和批次索引
        """

        if self.training:
            max = len(self.train_fnames)
        else:
            max = len(self.test_fnames)

        if self.file_point == max:
            if not self.training:
                # 文件指针到达末尾，并且当前是测试阶段，因此实验完成，抛出异常，让 main_eager.py 接管程序控制
                raise MyException('数据输送完成')

            # 文件指针到达末尾，当前是训练阶段，因此进入下一个 epoch
            self.epoch_index += 1
            self.file_point = 0

        if self.epoch_index > epoch:
            # 当完成了 epoch 次重复训练，epoch_index置为0，进入测试阶段
            self.epoch_index = 0  # 第0个epoch表示测试集
            self.file_point = 0
            max = len(self.test_fnames)
            self.training = False
            print('######################### 测试')

        # print('epoch={},point={}'.format(self.epoch_index, self.file_point))

        # 本 batch 的文件结束索引 = 当前文件指针位置 + batch大小
        end = self.file_point + batch_size

        # if end >= max:
        #     end = max

        x_data = []  # 训练数据
        y_data = []  # 训练标签，zero-filled list for 'one hot encoding'

        while self.file_point < end and self.file_point < max:
            # 遍历数据，从 file_point 开始，到 end 结束
            if self.training:
                imagePath = self.train_fnames[self.file_point]
            else:
                imagePath = self.test_fnames[self.file_point]
            try:
                # list.shape=[11, 80, 200] 这里可以换成其他任何读取单个样本的数据
                features = build_3d_input.get_features_3dmat(imagePath, window_size=self.width,
                                                             move_stride=self.move_stride, depth=self.depth)

            # 如果出现数据获取异常，则放弃该数据，获取下一个，为保持 batch_size 恒定， 让 end+1
            except EOFError:
                print('EOFError', imagePath)
                self.file_point += 1
                end += 1
                continue
            except MyException as e:
                # print(e.args)
                self.file_point += 1
                end += 1
                continue

            # 添加颜色通道，为数据增加一个维度
            features = np.expand_dims(features, axis=0)
            x_data.append(features)  # (image.data, dtype='float32')

            # 生成 one-hot 标签
            one_hot = np.zeros(int(self.num_class), dtype=np.int32)

            if self.training:
                one_hot[int(self.train_labs[self.file_point])] = 1
            else:
                one_hot[int(self.test_labs[self.file_point])] = 1

            y_data.append(one_hot)

            # 文件指针自增，获取下一个文件
            self.file_point += 1

        # print(np.shape(np.asarray(x_data, dtype=np.float32)))
        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32), self.epoch_index

