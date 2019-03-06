import librosa.display
from MyException import MyException


def get_features_3dmat(fileneme, depth, height, width):
    y, sr = librosa.load(fileneme, sr=None)

    features3d = stack_features(y, sr=sr, depth=depth, bands=height, frames=width)

    if len(features3d) < depth:
        # 时长： 10.5， len=8
        raise MyException('该数据时长不够：{}'.format(librosa.get_duration(filename=fileneme)))

    return features3d


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += int(window_size / 2)


def stack_features(y, sr, depth=5, bands=80, frames=200):
    window_size = 512 * (frames - 1)
    features3d = []

    for (start, end) in windows(y, window_size):
        # (1)此处是为了是将大小不一样的音频文件用大小window_size，
        # stride=window_size/2的窗口，分割为等大小的时间片段。
        # (2)计算每一个分割片段的log mel_sepctrogram.
        # 或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
        # 切割等大小的频谱图。
        if len(y[start:end]) == window_size:
            signal = y[start:end]
            features2d = cal_features(y=signal, sr=sr, height=bands)
            # print('111111', np.shape(features2d))
            # logspec = logspec.T.flatten()[:, np.newaxis].T
            features3d.append(features2d)

    return features3d[0:depth]


def cal_features(y, sr, height=80):
    # 语谱图 ,也叫时频域谱,最基本的物理特征
    # stft = librosa.core.stft(y, n_fft=1024, hop_length=512)
    # print('stft', stft.shape)

    # Mel频率倒谱系数
    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
    # print('mfccs: ', mfccs.shape)

    # 色度频率
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # print('chroma: ', chroma.shape)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=height, n_fft=1024, hop_length=512, power=2.0)
    logspec = librosa.amplitude_to_db(mel)
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

    # ext_features = np.r_[mfcc]

    # d = librosa.amplitude_to_db(mel)
    # librosa.display.specshow(d, y_axis='mel', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Power-Scaled Frequency Spectrogram')
    # plt.tight_layout()
    # plt.show()
    return logspec


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


