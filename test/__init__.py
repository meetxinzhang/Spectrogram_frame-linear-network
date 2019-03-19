# import librosa.display
# import numpy as np
# import scipy.misc
# import matplotlib.pyplot as plt
# import cv2
#
# conda install -c conda-forge librosa
# y, sr = librosa.load('D:/GitHub/ProjectX/test.mp3', sr=44100)
# t = len(y) / sr
# print(sr, t)
#
# # 语谱图 ,也叫时频域谱,最基本的物理特征 4 you  np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
# stft = librosa.core.stft(y, n_fft=1024, hop_length=512)
# stft = librosa.amplitude_to_db(stft)
# print('stft', stft.shape)
#
# Mel频率倒谱系数 laji
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
# mfcc = librosa.amplitude_to_db(mfcc)
# scipy.misc.imsave('D:/GitHub/ProjectX/test/mfcc.jpg', mfcc)
# print('mfccs: ', mfccs.shape)
#
# # 色度频率 laji
# # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
# # chroma = librosa.amplitude_to_db(chroma)
# # print('chroma: ', chroma.shape)
#
# #  xishu
"""
梅尔标度，the mel scale，由Stevens，Volkmann和Newman在1937年命名/
我们知道，频率的单位是赫兹（Hz），人耳能听到的频率范围是20-20000Hz，但人耳对Hz这种标度单位并不是线性感知关系。
例如如果我们适应了1000Hz的音调，如果把音调频率提高到2000Hz，我们的耳朵只能觉察到频率提高了一点点，根本察觉不到频率提高了一倍。
如果将普通的频率标度转化为梅尔频率标度
mel(f) = 2595*log10(1+f/700)
--------------------- 

原文：https://blog.csdn.net/qq_28006327/article/details/59129110 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""
# mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512, power=2.0)
# mel = librosa.amplitude_to_db(mel)
# mel_delta = librosa.feature.delta(mel)
# # scipy.misc.imsave('D:/GitHub/ProjectX/test/mel.jpg', mel)
# # # print('logsmel: ', logsmel.shape)
# #
# # # 计算光谱对比 laji
# # # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
# # # contrast = librosa.amplitude_to_db(contrast)
# # # print('contrast: ', contrast.shape)
# #
# # # 光谱质心 laji
# # # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
# # # print('tonnetz: ', tonnetz.shape)
# #
# # 节拍， start_bpm 速度估计器的初始猜测（每分钟节拍） tai shao
# tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, start_bpm=500, units='time')
#
# beat_mel_delta = librosa.util.sync(mel, beat_times)
# #
# # mohu
# # oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
# # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512)
# # tempogram = librosa.amplitude_to_db(tempogram)
#
# # 包络
# # librosa.feature.tempogram
#
#
# kernel = np.ones((3, 3), np.float32) / 25
# mel = cv2.filter2D(mel, -1, kernel)


# def windows(data, window_size):
#     start = 0
#     while start < len(data):
#         yield start, start + window_size
#         start += int(window_size / 2)
#
#
# def stack_features(y, sr, depth=10, bands=80, frames=200):
#     window_size = 512 * (frames - 1)
#     features3d = []
#
#     for (start, end) in windows(y, window_size):
#         # (1)此处是为了是将大小不一样的音频文件用大小window_size，
#         # stride=window_size/2的窗口，分割为等大小的时间片段。
#         # (2)计算每一个分割片段的log mel_sepctrogram.
#         # 或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
#         # 切割等大小的频谱图。
#         if len(y[start:end]) == window_size:
#             signal = y[start:end]
#             features2d = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=bands, n_fft=1024, hop_length=512, power=2.0 p)
#             features2d = librosa.amplitude_to_db(features2d)
#             # print('111111', np.shape(features2d))
#             # logspec = logspec.T.flatten()[:, np.newaxis].T
#
#             # blur = cv.bilateralFilter（img，9,75,75）
#             # kernel = np.ones((3, 3), np.float32) / 25
#             # features2d = cv2.filter2D(features2d, -1, kernel)
#             features3d.append(features2d)
#
#     return features3d[0:depth]
#
#
# features3d = stack_features(y, sr)
#
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b0.jpg', features3d[0])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b1.jpg', features3d[1])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b2.jpg', features3d[2])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b3.jpg', features3d[3])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b4.jpg', features3d[4])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b5.jpg', features3d[5])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b6.jpg', features3d[6])
# scipy.misc.imsave('D:/GitHub/ProjectX/test/b7.jpg', features3d[7])
# # scipy.misc.imsave('D:/GitHub/ProjectX/test/b8.jpg', features3d[8])
# # scipy.misc.imsave('D:/GitHub/ProjectX/test/b9.jpg', features3d[9])
#
# features3d = np.concatenate(features3d, axis=1)
# print(np.shape(features3d))
# scipy.misc.imsave('D:/GitHub/ProjectX/test/aa+.jpg', features3d)
#
# librosa.display.specshow(beat_mel_delta, y_axis='mel', fmax=22100, x_axis='time')
# plt.colorbar()
# plt.title('Mel Power-Scaled Frequency Spectrogram')
# plt.tight_layout()
# plt.show()

# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# with tf.device('/gpu:0'):
#     v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
#     v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
#     sumV12 = v1 + v2
#
#     with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#         print(sess.run(sumV12))


# # 新建一个 graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # 新建session with log_device_placement并设置为True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 运行这个 op.
# print(sess.run(c))


# import pywt
# cA, cD = pywt.dwt(y, 'db1')
# print(np.shape(cA), np.shape(cD))

# import tensorflow as tf
# import numpy as np
#
# x_rnn = [[[[1, 2],
#            [3, 4]],
#
#           [[1, 2],
#            [3, 4]]],
#
#
#          [[[1, 2],
#            [3, 4]],
#
#           [[1, 2],
#            [3, 4]]],
#
#
#          [[[1, 2],
#            [3, 4]],
#
#           [[1, 2],
#            [3, 4]]]]
#
# sess = tf.Session()
# # [3,2,2,2]
#
# x_rnn = tf.transpose(x_rnn, [0, 1, 3, 2])  # [3, 2, 2, 2]
# # print(sess.run(x_rnn))
#
# # shape = x_rnn.get_shape().as_list()
# # time_step = shape[1]
# # dim3 = shape[2] * shape[3]
# x_rnns = tf.unstack(x_rnn, axis=1)
# x_rnn = tf.concat(x_rnns, -1)
#
# print(sess.run(x_rnn))
