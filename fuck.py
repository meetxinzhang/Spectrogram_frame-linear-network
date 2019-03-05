# # A script to load images and make batch.
# # Dependency: 'nibabel' to load MRI (NIFTI) images
# # Reference: http://blog.naver.com/kjpark79/220783765651
#
import os
import numpy as np
import audio2mat
from class_names import class_names


class input_data(object):

    def __init__(self, train_file_dir):
        self.batch_index = 0
        self.file_point = 0
        self.filenames, self.labels = self.get_filenames(train_file_dir)

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
        a = list(temp[:, 0])
        b = list(temp[:, 1])

        # labels = [int(i) for i in labels]
        print("get the following labels ：")
        print(b)

        return a, b

    def next_batch(self, batch_size, num_class):
        max = len(self.filenames)

        end = self.file_point + batch_size

        if end >= max:
            end = max

        x_data = []
        y_data = []  # zero-filled list for 'one hot encoding'

        while self.file_point < end:
            imagePath = self.filenames[self.file_point]
            try:
                # [5, 80, 200]
                features = audio2mat.get_features_3dmat(imagePath)
            except EOFError:
                print('EOFError', imagePath)
                self.file_point += 1
                end += 1
                continue
            except Exception as e:
                print(e.args)
                self.file_point += 1
                end += 1
                continue

            # print(features.shape)
            # features = features[:, 0:2000]
            # features = np.pad(features, ((0, 0), (0, 2000 - len(features[1]))), mode='constant', constant_values=0)

            # 添加颜色通道
            features = np.expand_dims(features, axis=-1)
            print('fuck: 76 expand_dims', features.shape)
            # features = features.reshape([5, 80, 200, 1])

            # print('mfcc.shape paded: ', features.shape)
            x_data.append(features)  # (image.data, dtype='float32')

            one_hot = np.zeros(int(num_class), dtype=np.int32)
            one_hot[int(self.labels[self.file_point])] = 1

            y_data.append(one_hot)

            self.file_point += 1

        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32)

    # def next_batch(self, batch_size, num_class):
    #     max = len(self.filenames)
    #
    #     begin = self.batch_index
    #     end = self.batch_index + batch_size
    #
    #     if end >= max:
    #         end = max
    #         self.batch_index = 0
    #
    #     x_data = []
    #     y_data = np.zeros((batch_size, num_class))  # zero-filled list for 'one hot encoding'
    #     index = 0
    #
    #     for i in range(begin, end):
    #         imagePath = self.filenames[i]
    #         try:
    #             features = audio2mat.get_features_mat(imagePath)
    #         except EOFError:
    #             print('EOFError', imagePath)
    #
    #             continue
    #
    #         # print(features.shape)
    #         features = features[:, 0:100]
    #         features = np.pad(features, ((0, 0), (0, 100 - len(features[1]))), mode='constant', constant_values=0)
    #         # features = np.expand_dims(features, axis=2)
    #
    #         # print('mfcc.shape paded: ', features.shape)
    #
    #         x_data.append(features)  # (image.data, dtype='float32')
    #
    #         y_data[int(index)][int(self.labels[i])] = 1  # assign 1 to corresponding column (one hot encoding)
    #         index += 1
    #
    #     self.batch_index += batch_size  # update index for the next batch
    #     # x_data_ = x_data.reshape(batch_size, 92, 100)
    #
    #     return np.asarray(x_data, dtype=np.float32), y_data
