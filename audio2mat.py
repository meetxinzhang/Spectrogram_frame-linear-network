import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def get_features_mat(fileneme):
    y, sr = librosa.load(fileneme, sr=44100)

    return concatenate(y, sr)


def extract_feature(y, sr):
    # 语谱图 ,也叫时频域谱,最基本的物理特征
    # stft = librosa.core.stft(y, n_fft=1024, hop_length=512)
    # print('stft', stft.shape)

    # Mel频率倒谱系数
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
    # print('mfccs: ', mfccs.shape)

    # 色度频率
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # print('chroma: ', chroma.shape)

    # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512)
    # logsmel = librosa.feature.l
    # print('logsmel: ', logsmel.shape)

    # 计算光谱对比
    # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # print('contrast: ', contrast.shape)

    # 光谱质心
    # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    # print('tonnetz: ', tonnetz.shape)

    # 节拍， start_bpm 速度估计器的初始猜测（每分钟节拍）
    # tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, start_bpm=500, units='time')
    # print('beat_times', beat_times.shape)

    # 包络
    # librosa.feature.tempogram

    # d = librosa.amplitude_to_db(mel)
    # librosa.display.specshow(d, y_axis='mel', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Power-Scaled Frequency Spectrogram')
    # plt.tight_layout()
    # plt.show()

    return mfccs


# Concatenate all features and labels for each file
def concatenate(y, sr):
    mfccs = extract_feature(y, sr)
    ext_features = np.r_[mfccs]

    # print('ext_features', ext_features.shape)
    return ext_features


# def get_mp3_tus(fileneme_batch):
#     i = 0
#     img_s = []
#     for fileneme in fileneme_batch:
#         y, sr = librosa.load(fileneme.decode(), sr=44100)
#         # https://zhuanlan.zhihu.com/p/32292150
#         # E = D.real + (D.imag)*1j
#         ####################
#         img = concatenate(y, sr)
#         image = np.expand_dims(img, axis=2)
#         image = tf.image.resize_image_with_crop_or_pad(image, target_height=80, target_width=2000)
#         image = tf.cast(image, tf.float32)
#         image = tf.reshape(image, [80, 200, 10, 1])
#         img_s.append(image)
#
#         print("\r get features in this batch, progress : {}".format(i), end="")
#         i = i + 1
#
#     return img_s


