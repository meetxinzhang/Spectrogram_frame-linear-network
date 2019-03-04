"""
本地数据输入类
"""
import tensorflow as tf
import numpy as np
from class_names import class_names
import os
import audio2mat


class InputLocalData(object):

    def __init__(self, train_file_dir, num_epochs=None):
        # 训练和测试数据所在文件夹，/ 结尾
        self.train_file_dir = train_file_dir
        # 训练和测试数据 循环次数
        self.num_epochs = num_epochs
        # 文件名队列，详见 https://blog.csdn.net/dcrmg/article/details/79776876
        self.file_name_queue = self.get_files_name_queue()
    pass

    def get_files_name_queue(self):
        """
        获取文件名队列，放到内存里
        """
        img_list = []
        lab_list = []

        for train_class in os.listdir(self.train_file_dir):
            for dirs in os.listdir(self.train_file_dir + train_class):
                for file in os.listdir(self.train_file_dir + train_class + '/' + dirs):
                    img_list.append(self.train_file_dir + train_class + '/' + dirs + '/' + file)
                    lab = class_names.index(train_class)
                    lab_list.append(lab)
        temp = np.array([img_list, lab_list])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        # 随机打乱顺序
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])

        label_list = [int(i) for i in label_list]
        print("get the following labels ：")
        print(label_list)
        print(image_list)

        # """
        # 独热编码，这里其实没必要, 因为在计算交叉熵和准确率的时候又要变回来(training_graph.py第39,61行), 但为了学习下还是使用了.
        # :param indices: 待编码的下标数据,  is a scalar the output shape will be a vector of length
        # :param depth: 深度，指的是类别数
        # :param axis: 按行1列0方向
        # """
        # one_hot_label = tf.one_hot(indices=label_list, depth=self.num_class, axis=1, on_value=1, off_value=0, dtype=tf.int32)

        # convert the list of images and labels to tensor
        image_tensor = tf.cast(image_list, tf.string)
        label_tensor = tf.cast(label_list, tf.int32)
        # 这是创建 TensorFlow 的文件名队列，按照设定，每次从 [image_tensor, label_tensor] 列表中按顺序或者随机抽取出一个 tensor 放入文件名队列。
        # 详见 https://blog.csdn.net/dcrmg/article/details/79776876
        return tf.train.slice_input_producer([image_tensor, label_tensor], num_epochs=self.num_epochs)
    pass

    def get_batches(self, batch_size, capacity):
        """
        获取 图片和标签的 批次
        :param resize_w: 图片宽
        :param resize_h: 图片高
        :param batch_size: 每个批次里的图片数量
        :param capacity: 队列中的容量
        :return:
        """
        # 获取标签
        label = self.file_name_queue[1]
        filename = self.file_name_queue[0]
        # 获取图像
        image = tf.py_func(self._py_get_features, [filename], tf.uint8)
        # image = tf.image.resize_image_with_crop_or_pad(image, target_height=80, target_width=2000)
        image = tf.expand_dims(image, axis=2)
        image = tf.image.per_image_standardization(image)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [80, 200, 10, 1])
        print('image.shape paded: ', image.shape)

        # image_c = tf.read_file(self.file_name_queue[0])
        # 图像解码，不然得到的字符串
        # image = tf.image.decode_jpeg(image_c, channels=3)
        # 调整图像大小至 resize_w * resize_h，保持纵横比不变
        # tf.image.resize_images 不能保证图像的纵横比,这样用来做抓取位姿的识别,可能受到影响
        # image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        """
        标准化图像的像素值，加速模型的训练
        (x - mean) / adjusted_stddev
        其中x为RGB三通道像素值，mean分别为三通道像素的均值，
        adjusted_stddev = max(stddev, 1.0/sqrt(i mage.NumElements()))。
        stddev为三通道像素的标准差，image.NumElements()计算的是三通道各自的像素个数。
        """

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=10,
                                                  capacity=capacity,
                                                  allow_smaller_final_batch=True)
        # 转换像素值的类型 tf.float32
        image_batch2 = tf.cast(image_batch, dtype=tf.float32)
        # label_batch = tf.reshape(label_batch, [batch_size])

        return image_batch2, label_batch

    def _py_get_features(self, filename):
        mat = audio2mat.get_features_mat(filename.decode())

        mat = mat[:, 0:2000]
        print('mat_shape', np.shape(mat))
        return mat
