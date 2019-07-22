"""
这个文件是用来画图的
conda install -c conda-forge librosa

默认：Oriolus oriolus_240866
相似：pica 65795_12  anser 128217_2
"""
import librosa.display
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['figure.figsize'] = (15, 3)  # 设置figure_size尺寸


spe_data = [
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_240851.mp3', 0],  # jt0   Alternating song
    ['D:/GitHub/ProjectX/sounds_data/mp3/Pica+pica/Pica pica/Pica pica_61988.mp3', 0],  # xiangsi1 0  Pica pica
    # ['D:/GitHub/ProjectX/sounds_data/mp3/Anser+anser/Anser anser_128217.mp3', 2],  # xiangsi2  Anser anser
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_325361.mp3', 2],  # mjmc0000000
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_240851.mp3', 18],  # jt0   Alternating song
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_327400.mp3', 19],  # jt01 5  Alternating call
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_329582.mp3', 106],  # btmc0  Oriolus oriolus call and song
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_240866.mp3', 18],  # btmc 10  Low frequency noise and jt
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_253280.mp3', 1],  # dzaoy  Low frequency noise
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_372826.mp3', 0],  # gzaoy
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_270420.mp3', 8],  # cd   Overlapping song
    # ['D:\GitHub\ProjectX\sounds_data\mp3\Oriolus+oriolus\Oriolus oriolus_298307.mp3', 9],  # cd0 15  Overlapping call
]


def windows(data, window_size, move_stride):
    start = 0
    while start + window_size <= len(data):
        yield start, start + window_size
        start += move_stride


def stack_features(y, frames=600):
    window_size = 512 * (frames - 1)  # 因为两边会使用填充，所以窗口数目比帧长多1
    move_stride = int(window_size / 2)
    features3d = []

    row = len(y)
    for (start, end) in windows(y, window_size, move_stride):
        signal = y[start:end]
        features3d.append(signal)

    if window_size + (len(features3d) - 1) * move_stride < row:
        end = row
        start = end - window_size
        signal_last = y[start:end]
        features3d.append(signal_last)

    return features3d


# spe = spe_data[16]
for spe in spe_data:
    index = int(spe[1])
    y, sr = librosa.load(path=spe[0], sr=44100)
    t = len(y) / sr
    print(sr, t, index)

    signal = stack_features(y)[index]

    ################ 绘制波形图 ####################
    # fig = plt.figure(figsize=(12, 3))
    # ax1 = fig.add_subplot(111)
    # time = np.arange(0, len(signal)) * (1.0 / sr)
    # ax1.plot([i for i in time], [value for value in signal], 'b', label='Oriolus oriolus')
    # # 设置刻度字体大小
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # ax1.set_xlabel("Time", fontsize=18)
    # ax1.set_ylabel('Amplitude', fontsize=18)
    # plt.legend(loc='lower right', fontsize=18)
    ############################################

    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80, n_fft=1024, hop_length=512, power=2.0)
    logspec = librosa.amplitude_to_db(mel)
    logspec = logspec[:, 200:500]

    plt.figure()
    plt.subplot(2, 1, 1)

    librosa.display.specshow(logspec, y_axis='mel', fmax=44100, x_axis='frames')

    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Mel-Frequency')
    # plt.tight_layout()
    plt.show()
    plt.close()

# conda install -c conda-forge librosa
# y, sr = librosa.load('D:/GitHub/ProjectX/sounds_data/mp3/Oriolus+oriolus/Oriolus oriolus_240851.mp3', sr=44100)
# y = librosa.to_mono(y)
# t = len(y) / sr
# print(np.shape(y), t)
#
#
# def windows(data, window_size):
#     start = 0
#     while start < len(data):
#         yield start, start + window_size
#         start += int(window_size / 2)
#
#
# def stack_features(y, sr, bands=80, frames=600):
#
#     window_size = 512 * (frames - 1)  # 因为两边会使用填充，所以窗口数目比帧长多1
#     features3d = []
#     for (start, end) in windows(y, window_size):
#         # (1)此处是为了是将大小不一样的音频文件用大小window_size，
#         # stride=window_size/2的窗口，分割为等大小的时间片段。
#         # (2)计算每一个分割片段的log mel_sepctrogram.
#         # 或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
#         # 切割等大小的频谱图。
#         if len(y[start:end]) < window_size:
#             end = len(y)-1
#             start = end - window_size
#             if start < 0:
#                 break
#
#         signal = y[start:end]
#         mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=bands, n_fft=1024, hop_length=512, power=2.0)
#         logspec = librosa.amplitude_to_db(mel)
#
#         features3d.append(logspec)
#
#     return features3d
#
#
# mel = stack_features(y, sr)[0]
#
# plt.figure()
# plt.subplot(1, 1, 1)
# librosa.display.specshow(mel, y_axis='mel',  x_axis='frames', fmax=44100)  # x_axis='frames' / 'time'
# plt.xlabel('Time')
# # plt.ylabel('Mel scale')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# plt.close()
