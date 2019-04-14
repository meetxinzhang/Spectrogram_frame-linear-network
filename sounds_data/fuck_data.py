from PIL import Image
import psutil
import numpy as np
import os


def get_list():
    image_path = 'images'
    file_names = []

    for train_class in os.listdir(image_path):
        for pic in os.listdir(image_path + '/' + train_class):
            if os.path.isfile(image_path + '/' + train_class + '/' + pic):
                # save_path = 'new_images/' + train_class + '/' + pic.replace('.jpg', '')
                filename = image_path + '/' + train_class + '/' + pic

                file_names.append(filename)
    file_names.reverse()
    return file_names


def close_display_window(process_list):
    # for proc in psutil.process_iter():
    #     if proc not in process_list:
    #         proc.kill()
    for proc in psutil.process_iter():  # 遍历当前process
        if proc.name() == "Microsoft.Photos.exe":  # 如果process的name是display
            proc.kill()  # 关闭该pro


def windows(length, window_size):
    start = 0
    i = 0
    while start < length:
        yield start, start + window_size, i
        start += int(window_size*0.5)
        i += 1


def fuck_data(window_size=600):
    file_names = get_list()
    n = len(file_names)
    print('num of file:', n)

    if 'd' != input("按键说明：(小写)\n 使用W移动窗口，A上一张图片，D下一张图片，S保存剪切。\n 现在，输入D并回车开始处理下一张图片:"):
        exit(0)

    point = 0
    while point < n:

        image_path = file_names[point]
        save_dir = image_path.replace('images', 'new_images').replace('.jpg', '')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 读取图像
        old_img = np.asarray(Image.open(image_path))
        img = old_img.copy()
        img.setflags(write=True)

        for (start, end, i) in windows(np.shape(img)[1], window_size):
            if np.shape(img[:, start:end])[1] < window_size:
                break
            else:
                save_path = save_dir + '/' + str(i) + '.jpg'
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                else:
                    img[:, start:start+5] = 255  # 白色的竖线
                    img[:, end:end+5] = 255  # 白色的竖线
                    # 显示标记的图像
                    signal_show = Image.fromarray(img)

                    process_list = []
                    for proc in psutil.process_iter():
                        process_list.append(proc)
                    signal_show.show()
                    # for proc in psutil.process_iter():
                    #     if proc not in process_list:
                    #         print(proc.name())

                    ch = input('pic: ' + image_path + ', 输入操作:')
                    if ch == 's':  # 保存剪切
                        Image.fromarray(old_img[:, start:end]).save(save_path)
                        img[:, start:start+5] = 80  # 浅白色的竖线
                        img[:, end:end+5] = 80  # 浅白色的竖线
                        signal_show.close()
                        close_display_window(process_list)
                    elif ch == 'w':  # 移动窗口
                        img[:, start:start+5] = 80  # 浅白色的竖线
                        img[:, end:end+5] = 80  # 浅白色的竖线
                        close_display_window(process_list)
                        signal_show.close()
                        continue
                    elif ch == 'a':  # 上一张图片
                        point -= 2
                        close_display_window(process_list)
                        signal_show.close()
                        break
                    elif ch == 'd':  # 下一张图片
                        close_display_window(process_list)
                        signal_show.close()
                        break

        point += 1


if __name__ == '__main__':
    fuck_data(600)



