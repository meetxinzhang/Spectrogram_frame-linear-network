import os
import numpy as np
import audio2mat
from MyException import MyException
from class_names import class_names


class input_data(object):

    def __init__(self, train_file_dir, test_file_dir, depth=10, height=80, width=200, num_class=8):
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir
        self.epoch_index = 1
        self.file_point = 0

        self.depth = depth
        self.height = height
        self.width = width
        self.num_class = num_class
        self.filenames, self.labels = self.get_filenames(self.train_file_dir)

    def get_filenames(self, train_file_dir):
        filenames = []
        labels = []

        for train_class in os.listdir(train_file_dir):
            for dir in os.listdir(train_file_dir + '/' + train_class):
                for pic in os.listdir(train_file_dir + '/' + train_class + '/' + dir):
                    if os.path.isfile(train_file_dir + '/' + train_class + '/' + dir + '/' + pic):
                        filenames.append(train_file_dir + '/' + train_class + '/' + dir + '/' + pic)
                        label = class_names.index(train_class)
                        labels.append(int(label))

        temp = np.array([filenames, labels])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        # 随机打乱顺序
        np.random.shuffle(temp)
        file_list = list(temp[:, 0])
        lab_list = list(temp[:, 1])

        # labels = [int(i) for i in labels]
        print("数据 ：", len(file_list))
        print("标签 ：", len(lab_list))

        return file_list, lab_list

    def get_test_filenames(self, test_file_dir):
        filenames = []
        labels = []
        for train_class in os.listdir(test_file_dir):
                for pic in os.listdir(test_file_dir + '/' + train_class):
                    if os.path.isfile(test_file_dir + '/' + train_class + '/' + pic):
                        filenames.append(test_file_dir + '/' + train_class + '/' + pic)
                        label = class_names.index(train_class)
                        labels.append(int(label))

        temp = np.array([filenames, labels])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        # 随机打乱顺序
        np.random.shuffle(temp)
        file_list = list(temp[:, 0])
        lab_list = list(temp[:, 1])

        # labels = [int(i) for i in labels]
        print("测试数据 ：", len(file_list))
        print("测试标签 ：", len(lab_list))

        return file_list, lab_list

    def next_batch(self, batch_size, epoch=1):
        max = len(self.filenames)

        if self.epoch_index > epoch:
            print('######################### 测试')
            self.epoch_index = 1
            self.file_point = 0
            self.filenames, self.labels = self.get_test_filenames(self.test_file_dir)

        if self.file_point == max:
            self.epoch_index += 1
            self.file_point = 0
            self.filenames, self.labels = self.get_filenames(self.train_file_dir)
            print('############################# \n epoch=', self.epoch_index)

        print('next_batch', self.file_point)

        end = self.file_point + batch_size

        # if end >= max:
        #     end = max

        x_data = []
        y_data = []  # zero-filled list for 'one hot encoding'

        while self.file_point < end and self.file_point < max:
            # print('progress:{} and batch_end={}'.format(self.file_point, end), end="\r")
            imagePath = self.filenames[self.file_point]
            try:
                # [10, 80, 200] 这里可以换成其他任何读取单个样本的数据
                features = audio2mat.get_features_3dmat(
                    imagePath, depth=self.depth, height=self.height, width=self.width)
            except EOFError:
                print('EOFError', imagePath)
                self.file_point += 1
                end += 1
                continue
            except MyException as e:
                print(e.args)
                self.file_point += 1
                end += 1
                continue

            # print(features.shape)
            # features = features[:, 0:2000]
            # features = np.pad(features, ((0, 0), (0, 2000 - len(features[1]))), mode='constant', constant_values=0)

            # 添加颜色通道
            features = np.expand_dims(features, axis=-1)
            # features = features.reshape([5, 80, 200, 1])

            # print('mfcc.shape paded: ', features.shape)
            x_data.append(features)  # (image.data, dtype='float32')

            one_hot = np.zeros(int(self.num_class), dtype=np.int32)
            one_hot[int(self.labels[self.file_point])] = 1

            y_data.append(one_hot)

            self.file_point += 1

        # print(np.shape(np.asarray(x_data, dtype=np.float32)))
        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32), self.epoch_index

