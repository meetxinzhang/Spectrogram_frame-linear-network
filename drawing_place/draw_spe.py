'''
这个文件是用来画图的
'''
import librosa.display
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import cv2
#
# conda install -c conda-forge librosa
y, sr = librosa.load('D:/GitHub/ProjectX/sounds_data/mp3/Oriolus+oriolus/Oriolus oriolus_240866.mp3', sr=44100)
t = len(y) / sr
print(sr, t)


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


signal = stack_features(y)[45]
mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80, n_fft=1024, hop_length=512, power=2.0)
logspec = librosa.amplitude_to_db(mel)


plt.figure()
plt.subplot(3, 1, 1)

librosa.display.specshow(logspec, y_axis='mel', fmax=44100, x_axis='time')

# fig = plt.figure(figsize=(10, 4))
# ax1 = fig.add_subplot(111)
# time = np.arange(0, len(signal)) * (1.0 / sr)
# ax1.plot([i for i in time], [value for value in signal], 'b', label='Oriolus oriolus')
# # 设置刻度字体大小
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# ax1.set_xlabel("Time(s)", fontsize=18)
# ax1.set_ylabel('Amplitude(mm)', fontsize=18)
# plt.legend(loc='lower right', fontsize=18)


plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Mel scale(mel)')
# plt.tight_layout()
plt.show()

# pica 65795_12     anser 128217_2

# # conda install -c conda-forge librosa
# y, sr = librosa.load('D:/GitHub/ProjectX/sounds_data/mp3/Oriolus+oriolus/Oriolus oriolus_240851.mp3', sr=44100)
# t = len(y) / sr
# print(sr, t)
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
# plt.subplot(2, 1, 1)
# librosa.display.specshow(mel[:, 400:600], y_axis='mel', fmax=22100, x_axis='time')
#
# plt.colorbar()
# plt.tight_layout()
# plt.show()