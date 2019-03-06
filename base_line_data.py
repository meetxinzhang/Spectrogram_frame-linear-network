# # A script to load images and make batch.
# # Dependency: 'nibabel' to load MRI (NIFTI) images
# # Reference: http://blog.naver.com/kjpark79/220783765651
#
import os
import numpy as np
import librosa.display
from class_names import class_names
from MyException import MyException
import matplotlib.pyplot as plt


class base_line_data(object):

    def __init__(self, train_file_dir, height=512, width=900, num_class=8):
        self.batch_index = 0
        self.file_point = 0

        self.filenames, self.labels = self.get_filenames(train_file_dir)
        self.height = height
        self.width = width
        self.num_class = num_class

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
        print("总共有数据 ：", len(lab_list))

        return file_list, lab_list

    def next_batch(self, batch_size):
        print('next_batch', self.file_point)

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
                y, sr = librosa.load(imagePath, sr=None)
                # 语谱图 ,也叫时频域谱,最基本的物理特征 4 you  np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.height, n_fft=1024, hop_length=512, power=2.0)
                logspec = librosa.amplitude_to_db(mel)
                logspec = np.asarray(logspec, dtype=np.float32)

                if logspec.shape[1] <= 900:
                    raise MyException('该数据时长不够')
                else:
                    logspec = logspec[0:self.height, 0:self.width]
                    # librosa.display.specshow(logspec, y_axis='chroma', x_axis='time')
                    # plt.colorbar()
                    # plt.title('Chromagram')
                    # plt.tight_layout()
                    # plt.show()

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
            features = np.expand_dims(logspec, axis=-1)
            # features = features.reshape([512, 900, 1])

            # print('mfcc.shape paded: ', features.shape)
            x_data.append(features)  # (image.data, dtype='float32')

            one_hot = np.zeros(int(self.num_class), dtype=np.int32)
            one_hot[int(self.labels[self.file_point])] = 1

            print('22222', one_hot)

            y_data.append(one_hot)

            self.file_point += 1

        print(np.shape(np.asarray(x_data, dtype=np.float32)))
        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32)


if __name__ == '__main__':
    path = 'D:/GitHub/ProjectX/sounds/Accipiter+gentilis/Accipiter gentilis'
    data = base_line_data(train_file_dir=path, height=400, width=800, num_class=8)
    for i in range(50):
        batch_x, batch_y = data.next_batch(batch_size=5)
