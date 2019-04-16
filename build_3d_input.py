import librosa.display
from MyException import MyException
import numpy as np
from PIL import Image

import scipy.misc
import random
import cv2

#
# def get_features_3dmat(fileneme, depth, height, width, training=True):
#     y, sr = librosa.load(fileneme, sr=None)
#     # if len(y)/sr < 2:
#     #     raise MyException('时长太短；{}')
#
#     features3d = stack_features(y, sr=sr, depth=depth, bands=height, frames=width)
#
#     if len(features3d) == 0:
#         raise MyException('depth==0')
#
#     # 填充
#     if len(features3d) < depth:
#         for i in range(depth-len(features3d)):
#             piece_add = features3d[i]
#             features3d.append(piece_add)
#
#     # len_feat = len(features3d)
#     # if len_feat < depth:
#     #     # 时长： 10.5， len=8
#     #     raise MyException('该数据时长不够：{}'.format(librosa.get_duration(filename=fileneme)))
#
#     if training:
#         # 数据增强2 - 模拟队列数据结构，左平移每个特征图
#         seed_move = random.randint(0, 1)
#         for i in range(seed_move):
#             temp = features3d.pop(0)
#             features3d.append(temp)
#
#     return features3d
#
#
# def windows(data, window_size):
#     start = 0
#     while start < len(data):
#         yield start, start + window_size
#         start += int(window_size / 2)
#
#
# def stack_features(y, sr, depth=5, bands=80, frames=200):
#     window_size = 512 * (frames - 1)  # 因为两边会使用填充，所以窗口数目比帧长多1
#     features3d = []
#     seed_if = random.randint(0, 1)
#     for (start, end) in windows(y, window_size):
#         # (1)此处是为了是将大小不一样的音频文件用大小window_size，
#         # stride=window_size/2的窗口，分割为等大小的时间片段。
#         # (2)计算每一个分割片段的log mel_sepctrogram.
#         # 或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
#         # 切割等大小的频谱图。
#         if len(y[start:end]) == window_size:
#             signal = y[start:end]
#             features2d = cal_features(y=signal, sr=sr, height=bands)
#             # print('111111', np.shape(features2d))
#             # logspec = logspec.T.flatten()[:, np.newaxis].T
#
#             # blur = cv.bilateralFilter（img，9,75,75）
#             kernel = np.ones((3, 3), np.float32) / 25
#             features2d = cv2.filter2D(features2d, -1, kernel)
#
#             # 数据增强1 - 垂直翻转
#             if seed_if == 0:
#                 features2d = np.flipud(features2d)
#
#             features3d.append(features2d)
#
#     return features3d[0:depth]
#
#
# def cal_features(y, sr, height=80):
#     # 语谱图 ,也叫时频域谱,最基本的物理特征
#     # stft = librosa.core.stft(y, n_fft=1024, hop_length=512)
#     # print('stft', stft.shape)
#
#     # Mel频率倒谱系数
#     # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
#     # print('mfccs: ', mfccs.shape)
#
#     # 色度频率
#     # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     # print('chroma: ', chroma.shape)
#
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=height, n_fft=1024, hop_length=512, power=2.0)
#     logspec = librosa.amplitude_to_db(mel)
#     # print('logsmel: ', logsmel.shape)
#
#     # 计算光谱对比
#     # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     # print('contrast: ', contrast.shape)
#
#     # 光谱质心
#     # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
#     # print('tonnetz: ', tonnetz.shape)
#
#     # 节拍， start_bpm 速度估计器的初始猜测（每分钟节拍）
#     # tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, start_bpm=500, units='time')
#     # print('beat_times', beat_times.shape)
#
#     # 包络
#     # librosa.feature.tempogram
#
#     # ext_features = np.r_[mfcc]
#
#     # d = librosa.amplitude_to_db(mel)
#     # librosa.display.specshow(d, y_axis='mel', x_axis='time')
#     # plt.colorbar(format='%+2.0f dB')
#     # plt.title('Mel Power-Scaled Frequency Spectrogram')
#     # plt.tight_layout()
#     # plt.show()
#     return logspec


def get_features_3dmat(fileneme, depth, height, width, training=True):
    # y, sr = librosa.load(fileneme, sr=None)
    #
    # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=height, n_fft=1024, hop_length=512, power=2.0)
    # logspec = librosa.amplitude_to_db(mel)
    #
    # logspec = np.asarray(scipy.misc.toimage(logspec))
    logspec = np.asarray(Image.open(fileneme))

    features3d = pick_feat(logspec, depth=depth)

    len_feat = len(features3d)
    if len_feat < depth:
        # 时长： 10.5， len=8
        raise MyException('该数据时长不够：{}'.format(np.shape(logspec)))

    return features3d


def windows(length, window_size):
    start = 0
    while start < length:
        yield start, start + window_size
        start += int(window_size*0.5)


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


# if __name__ == '__main__':
#     a = get_features_3dmat('sounds_data//Oriolus+oriolus/Oriolus oriolus/Oriolus oriolus_430350.mp3', depth=5, height=80, width=200)
#     scipy.misc.imsave('D:/GitHub/ProjectX/drawing_place/in.jpg', a)
