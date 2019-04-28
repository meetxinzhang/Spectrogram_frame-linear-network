import os
import numpy as np
import librosa.display
import scipy.misc
from MyException import MyException


def file2img(file_dir):
    for train_class in os.listdir(file_dir):
        for pic in os.listdir(file_dir + '/' + train_class):
            if os.path.isfile(file_dir + '/' + train_class + '/' + pic):
                save_path = 'images/' + train_class + '/' + pic.replace('.mp3', '') + '.jpg'
                filename = file_dir + '/' + train_class + '/' + pic

                if not os.path.exists('images/' + train_class):
                    os.makedirs('images/' + train_class)
                if os.path.exists(save_path):
                    print('pass')
                    continue
                else:
                    y, sr = librosa.load(filename, sr=None)
                    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512, power=2.0)
                    logspec = librosa.amplitude_to_db(mel)

                    try:
                        scipy.misc.imsave(save_path, logspec[:, 0:65500])
                        print('save: ', save_path)
                    except OSError as e:
                        print(logspec.shape)
                        print(e.args)


imagePath = 'mp3/Pica+pica'
file2img(imagePath)
pass
