import os
import numpy as np
import build_3d_input
from MyException import MyException
from class_names import class_names


class input_data(object):

    def __init__(self, file_dir, move_stride=100, depth=10, num_class=4):
        self.file_dir = file_dir
        self.training = True
        self.epoch_index = 1  # epoch次数指针，训练从1开始
        self.file_point = 0  # 第0个epoch表示测试集

        self.depth = depth
        self.move_stride = move_stride
        self.num_class = num_class

        self.train_fnames, self.train_labs, self.test_fnames, self.test_labs\
            = self.get_filenames(self.file_dir)

    def get_filenames(self, file_dir):
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

        if self.training:
            max = len(self.train_fnames)
        else:
            max = len(self.test_fnames)

        if self.file_point == max:
            if not self.training:
                raise MyException('数据输送完成')

            self.epoch_index += 1
            self.file_point = 0

        if self.epoch_index > epoch:
            self.epoch_index = 0  # 第0个epoch表示测试集
            self.file_point = 0
            max = len(self.test_fnames)
            self.training = False
            print('######################### 测试')

        # print('epoch={},point={}'.format(self.epoch_index, self.file_point))

        end = self.file_point + batch_size

        # if end >= max:
        #     end = max

        x_data = []
        y_data = []  # zero-filled list for 'one hot encoding'

        while self.file_point < end and self.file_point < max:
            # ##########数据##############
            if self.training:
                imagePath = self.train_fnames[self.file_point]
            else:
                imagePath = self.test_fnames[self.file_point]
            try:
                # list.shape=[5, 80, 200] 这里可以换成其他任何读取单个样本的数据
                features = build_3d_input.get_features_3dmat(imagePath, move_stride=self.move_stride, depth=self.depth)
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

            # 添加颜色通道
            features = np.expand_dims(features, axis=-1)
            x_data.append(features)  # (image.data, dtype='float32')

            # ##########标签##############
            one_hot = np.zeros(int(self.num_class), dtype=np.int32)

            if self.training:
                one_hot[int(self.train_labs[self.file_point])] = 1
            else:
                one_hot[int(self.test_labs[self.file_point])] = 1

            y_data.append(one_hot)

            self.file_point += 1

        # print(np.shape(np.asarray(x_data, dtype=np.float32)))
        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32), self.epoch_index

