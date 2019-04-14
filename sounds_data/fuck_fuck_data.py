from PIL import Image
import numpy as np
import os


def windows(length, window_size):
    start = 0
    i = 0
    while start < length:
        yield start, start + window_size, i
        start += int(window_size*0.5)
        i += 1


def fuck_data(window_size=600):

    files_path = 'images'

    for train_class in os.listdir(files_path):
        if train_class != 'Pica+pica':
            continue
        for pic_name in os.listdir(files_path + '/' + train_class):
            if os.path.isfile(files_path + '/' + train_class + '/' + pic_name):

                filename = files_path + '/' + train_class + '/' + pic_name

                save_dir = 'new_images' + '/' + train_class

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 读取图像
                old_img = np.asarray(Image.open(filename))
                img = old_img.copy()
                img.setflags(write=True)

                for (start, end, i) in windows(np.shape(img)[1], window_size):
                    if np.shape(img[:, start:end])[1] < window_size:
                        end = np.shape(img)[1]
                        start = end - window_size
                        if start < 0:
                            break

                    save_path = save_dir + '/' + pic_name.replace('.jpg', '(' + str(i) + ')') + '.jpg'
                    if os.path.exists(save_path):
                        print('--exist: ', save_path)
                        continue
                    else:
                        Image.fromarray(old_img[:, start:end]).save(save_path)
                        print('save:', save_path)


if __name__ == '__main__':
    fuck_data(600)



