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
