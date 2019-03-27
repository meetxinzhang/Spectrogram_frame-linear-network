import os
import numpy as np
import librosa.display
import scipy.misc
from MyException import MyException


# def windows(length, window_size):
#     start = 0
#     while start < length:
#         yield start, start + window_size
#         start += int(window_size*0.5)
#
#
# def pick_feat(mat):
#
#     window_size = 200
#     max = 0
#     best_index = 0
#
#     for (start, end) in windows(np.shape(mat)[1], window_size):
#         if np.shape(mat[:, start:end])[1] == window_size:
#             signal = mat[10:-1, start:end]
#             he = sum(sum(signal))
#             if max <= he:
#                 max = he
#                 best_index = start
#
#     feature = mat[:, best_index:best_index + window_size]
#
#     return feature
#
#
# def file2img(file_dir):
#
#     for train_class in os.listdir(file_dir):
#         for dir in os.listdir(file_dir + '/' + train_class):
#             for pic in os.listdir(file_dir + '/' + train_class + '/' + dir):
#                 if os.path.isfile(file_dir + '/' + train_class + '/' + dir + '/' + pic):
#
#                     save_path = 'single_images/' + train_class + '/' + '/' + pic.replace('.mp3', '') + '.jpg'
#                     filename = file_dir + '/' + train_class + '/' + dir + '/' + pic
#
#                     if os.path.exists(save_path):
#                         print('pass')
#                         continue
#                     else:
#                         y, sr = librosa.load(filename, sr=44100)
#                         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512,
#                                                              power=2.0)
#                         logspec = librosa.amplitude_to_db(mel)
#                         logspec = np.asarray(scipy.misc.toimage(logspec))
#
#                         if np.shape(logspec)[1] < 200:
#                             print(' '+str(librosa.get_duration(filename=filename))+', '+filename)
#                             continue
#                         feat = pick_feat(logspec)
#
#                         scipy.misc.imsave(save_path, feat)
#                         print('save: ', save_path)
#
#
# imagePath = 'D:/GitHub/ProjectX/sounds'
# file2img(imagePath)

def file2img(file_dir):

    for train_class in os.listdir(file_dir):
        for dir in os.listdir(file_dir + '/' + train_class):
            for pic in os.listdir(file_dir + '/' + train_class + '/' + dir):
                if os.path.isfile(file_dir + '/' + train_class + '/' + dir + '/' + pic):

                    save_path = 'images/' + train_class + '/' + pic.replace('.mp3', '') + '.jpg'
                    filename = file_dir + '/' + train_class + '/' + dir + '/' + pic

                    if os.path.exists(save_path):
                        print('pass')
                        continue
                    else:
                        y, sr = librosa.load(filename, sr=None)
                        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512,
                                                             power=2.0)
                        logspec = librosa.amplitude_to_db(mel)
                        # logspec = np.asarray(scipy.misc.toimage(logspec))

                        try:
                            scipy.misc.imsave(save_path, logspec[:, 0:65500])
                            print('save: ', save_path)
                        except OSError as e:
                            print(logspec.shape)
                            print(e.args)


imagePath = 'D:/GitHub/ProjectX/sounds'
file2img(imagePath)
