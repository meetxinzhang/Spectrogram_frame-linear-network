import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = 'test.mp3'
y, sr = librosa.load(file, sr=None)
print(sr)


def plotSpectrogram(y, sr):
    # Plot the Mel power-scaled frequency spectrum, with any factor of 128 frequency bins and 512 frames (frame default)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512,
                                         spec_fmin=500, spec_fmax=15000)
    print('mel', mel.shape)
    d = librosa.amplitude_to_db(mel)
    print('d', d.shape)
    librosa.display.specshow(d, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Power-Scaled Frequency Spectrogram')
    plt.tight_layout()
    plt.show()
    return mel


def extract_feature(sound):
    stft = librosa.core.stft(sound)
    print(stft.shape)
    mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40).T
    print(mfccs.shape)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr).T
    print(chroma.shape)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=512).T
    print(mel.shape)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr).T
    print(contrast.shape)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sr).T
    print(tonnetz.shape)
    return mfccs, chroma, mel, contrast, tonnetz


# Concatenate all features and labels for each file
def concatenate(sound):
    features = np.empty((0, 193))
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(sound)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])

    librosa.display.specshow(features, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Power-Scaled Frequency Spectrogram')
    plt.tight_layout()
    plt.show()
    return np.array(features)


concatenate(y)
