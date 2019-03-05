# import librosa
# import numpy as np
#
# print('时长：', librosa.get_duration(filename='D:/GitHub/ProjectX/test.mp3'))
# y, sr = librosa.load('D:/GitHub/ProjectX/test.mp3', sr=None)
# stft = librosa.stft(y)
# print('stft', np.shape(stft))
#
# # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512)
# # logspec = librosa.amplitude_to_db(mel)
# # print('logspec', np.shape(logspec))
#
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
# print('mfcc', np.shape(mfcc))
#
# d = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
# print('d', np.shape(d))
#
#
# def windows(data, window_size):
#     start = 0
#     while start < len(data):
#         yield start, start + window_size
#         start += int(window_size / 2)
#
#
# def extract_features(y, bands=80, frames=200):
#     window_size = 512 * (frames - 1)
#     log_specgrams = []
#
#     for (start, end) in windows(y, window_size):
#         # (1)此处是为了是将大小不一样的音频文件用大小window_size，
#         # stride=window_size/2的窗口，分割为等大小的时间片段。
#         # (2)计算每一个分割片段的log mel_sepctrogram.
#         # 或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
#         # 切割等大小的频谱图。
#         if len(y[start:end]) == window_size:
#             signal = y[start:end]
#             mel = librosa.feature.melspectrogram(signal, n_mels=bands, n_fft=1024, hop_length=512)
#             logspec = librosa.amplitude_to_db(mel)
#             print('111111', np.shape(logspec))
#             # logspec = logspec.T.flatten()[:, np.newaxis].T
#             log_specgrams.append(logspec)
#
#     return log_specgrams[0:10]
#
#
# logs = extract_features(y)
# print('logs', np.shape(logs))
# print(len(logs))
#
# filenames = [1,2,3,4,5,6,7,8,9]
# labels = [1,2,3,4,5,6,7,8,9,]
#
# print(filenames)
# temp = np.array([filenames, labels])
# # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
# print(temp)
# temp = temp.transpose()
# print(temp)
# # 随机打乱顺序
# np.random.shuffle(temp)
# a = list(temp[:, 0])
# b = list(temp[:, 1])
#
# print(a)
# print(b)