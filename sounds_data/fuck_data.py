from PIL import Image
import numpy as np
import os
import scipy.misc


def get_list():
    image_path = 'images'
    file_names = []

    for train_class in os.listdir(image_path):
        for pic in os.listdir(image_path + '/' + train_class):
            if os.path.isfile(image_path + '/' + train_class + '/' + pic):
                # save_path = 'new_images/' + train_class + '/' + pic.replace('.jpg', '')
                filename = image_path + '/' + train_class + '/' + pic

                file_names.append(filename)
    return file_names

    # if os.path.exists(save_path):
    #     print('pass')
    #     continue
    # else:
    #     img = np.asarray(Image.open(image_path))
    #
    #
    #     try:
    #         scipy.misc.imsave(save_path, logspec[:, 0:65500])
    #         print('save: ', save_path)
    #     except OSError as e:
    #         print(logspec.shape)
    #         print(e.args)


def wait_keyb_in(condition, out_str=''):
    str = input(out_str)
    if str == condition:
        return True
    else:
        return False


def windows(length, window_size):
    start = 0
    i = 0
    while start < length:
        yield start, start + window_size, i
        start += int(window_size*0.5)
        i += 1


def pick_feat(mat, depth=5):
    window_size = 200
    max = 0
    best_index = 0
    features3d = []

    for (start, end) in windows(np.shape(mat)[1], window_size):
        if np.shape(mat[:, start:end])[1] == window_size:
            signal = mat[10:-1, start:end]
            he = sum(sum(signal))
            if max <= he:
                max = he
                best_index = start

    index_1th = best_index + int(window_size*0.5*(0-2))
    if index_1th + window_size*3 > np.shape(mat)[1]:
        index_1th = np.shape(mat)[1] - window_size*3
    if index_1th < 0:
        index_1th = 0

    for i in range(depth):
        feature = mat[:, index_1th:index_1th+window_size]
        if np.shape(feature)[1] < window_size:
            break
        features3d.append(feature)
        index_1th += int(window_size*0.5)

    return features3d


def fuck_data(window_size=600):
    file_names = get_list()
    n = len(file_names)
    print('num of file:', n)

    if not wait_keyb_in('f', "使用AD操作方向，F下一张图片，S保存剪切。\n 现在，按F开始(小写)"):
        exit(0)

    point = 0
    image_path = file_names[point]
    img = np.asarray(Image.open(image_path))

    for (start, end, i) in windows(np.shape(img)[1], window_size):
        if np.shape(img[:, start:end])[1] < window_size:
            break
        else:
            signal = img
            signal[:, start] = 0  # 白色的竖线
            signal[:, end] = 0  # 白色的竖线

            save_path = image_path.replace('images', 'new_images').replace('.jpg', '('+str(i)+')'+'.jpg')
            Image.fromarray(signal).show()

            if wait_keyb_in('s', '输入操作'):
                Image.fromarray(img[:, start:end]).save(save_path)
            else:
                



