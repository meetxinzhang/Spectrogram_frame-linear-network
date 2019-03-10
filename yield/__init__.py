# import os
# import numpy as np
# import librosa.display
# import scipy.misc
#
#
# def file2img(train_file_dir):
#
#     for train_class in os.listdir(train_file_dir):
#         for dir in os.listdir(train_file_dir + '/' + train_class):
#             for pic in os.listdir(train_file_dir + '/' + train_class + '/' + dir):
#                 if os.path.isfile(train_file_dir + '/' + train_class + '/' + dir + '/' + pic):
#
#                     save_path =           'images/' + train_class + '/' + dir + '/' + pic.replace('.mp3', '') + '.jpg'
#                     filename = train_file_dir + '/' + train_class + '/' + dir + '/' + pic
#
#                     if os.path.exists(save_path):
#                         print('pass')
#                         continue
#                     else:
#                         y, sr = librosa.load(filename, sr=None)
#                         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512,
#                                                              power=2.0)
#                         mel = librosa.amplitude_to_db(mel)
#
#                         print(np.shape(mel))
#                         try:
#                             scipy.misc.imsave(save_path, mel)
#                         except Exception as e:
#                             print(e.args)
#                             print(filename)
#                             continue
#
#
# imagePath = 'D:/GitHub/ProjectX/sounds'
# file2img(imagePath)
