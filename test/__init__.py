# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# y, sr = librosa.load('D:/GitHub/ProjectX/test.mp3', sr=None)
#
# # 语谱图 ,也叫时频域谱,最基本的物理特征 4 you  np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
# stft = librosa.core.stft(y, n_fft=1024, hop_length=512)
# stft = librosa.amplitude_to_db(stft)
# print('stft', stft.shape)
#
# # Mel频率倒谱系数 laji
# # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
# # mfcc = librosa.amplitude_to_db(mfcc)
# # print('mfccs: ', mfccs.shape)
#
# # 色度频率 laji
# # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
# # chroma = librosa.amplitude_to_db(chroma)
# # print('chroma: ', chroma.shape)
#
# #  xishu
# mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512, power=2.0)
# mel = librosa.amplitude_to_db(mel)
# # print('logsmel: ', logsmel.shape)
#
# # 计算光谱对比 laji
# # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
# # contrast = librosa.amplitude_to_db(contrast)
# # print('contrast: ', contrast.shape)
#
# # 光谱质心 laji
# # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
# # print('tonnetz: ', tonnetz.shape)
#
# # 节拍， start_bpm 速度估计器的初始猜测（每分钟节拍） tai shao
# # tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, start_bpm=500, units='time')
# # print('beat_times', beat_times.shape)
#
# # mohu
# # oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
# # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512)
# # tempogram = librosa.amplitude_to_db(tempogram)
#
# # 包络
# # librosa.feature.tempogram
#
#
# librosa.display.specshow(stft, y_axis='chroma', x_axis='time')
# plt.colorbar()
# plt.title('Chromagram')
# plt.tight_layout()
# plt.show()
