import os
import tensorflow as tf
import numpy as np
from audio2mat import get_features_3dmat
from class_names import class_names


def get_files_names(train_file_dir):
    """
    获取文件名队列，放到内存里
    """
    img_list = []
    lab_list = []

    for train_class in os.listdir(train_file_dir):
        for dir in os.listdir(train_file_dir + train_class):
            for pic in os.listdir(train_file_dir + train_class + '/' + dir):
                img_list.append(train_file_dir + train_class + '/' + dir + '/' + pic)

                lab = class_names.index(train_class)
                lab_list.append(int(lab))

    return img_list, lab_list


def _py_func_filter(filename, label):
    try:
        get_features_3dmat(filename.decode(), depth=5, height=80, width=200)
        return True
    except EOFError:
        print('EOFError', filename.decode())
        return False
    except Exception as e:
        print(print("Unexpected Error: {}".format(e)))
        return False


def _py_func_map(filename, label):
    image = get_features_3dmat(filename.decode(), depth=5, height=80, width=200)

    image = np.expand_dims(image, axis=2)
    image = tf.image.resize_image_with_crop_or_pad(image, target_height=80, target_width=2000)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [80, 200, 10, 1])
    print('image.shape paded: ', image.shape)

    return image, label


def get_dataset(train_file_dir, batch_size):
    img_train, lab_train = get_files_names(train_file_dir)
    n = len(lab_train)
    len_test = int(n/10)
    print('总共 {} 份样本'.format(n))
    print('其中 {} 份用于测试'.format(len_test))
    img_test = []
    lab_test = []
    for i in range(int(len_test)):
        r = np.random.randint(0, len(lab_train))
        lab_test.append(lab_train[r])
        img_test.append(img_train[r])

        lab_train.remove(lab_train[r])
        img_train.remove(img_train[r])

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (img_train, lab_train)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (img_test, lab_test)
    )

    print('checkout dataset...')
    train_dataset = train_dataset.filter(lambda file, lab: tf.py_func(
        _py_func_filter, [file, lab], bool
    ))
    test_dataset = test_dataset.filter(lambda file, lab: tf.py_func(
        _py_func_filter, [file, lab], bool
    ))

    # train_dataset = train_dataset.apply(
    #     tf.contrib.data.map_and_batch(
    #         map_func=lambda file, lab: tuple(tf.py_func(self._py_func_map, [file, lab], [tf.float32, lab.dtype])),
    #         batch_size=batch_size)).cache(filename='cache_train')
    # test_dataset = test_dataset.apply(
    #     tf.contrib.data.map_and_batch(
    #         map_func=lambda file, lab: tuple(tf.py_func(self._py_func_map, [file, lab], [tf.float32, lab.dtype])),
    #         batch_size=batch_size)).cache(filename='cache_test')

    print('shuffling...')
    train_dataset = train_dataset.shuffle(buffer_size=10).repeat(10).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=10).repeat(10).batch(batch_size)

    return train_dataset, test_dataset
