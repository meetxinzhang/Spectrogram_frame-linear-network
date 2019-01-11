import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = 'test.mp3'
y, sr = librosa.load(file, sr=None)
print(sr)

def plotSpectrogram(y, sr):
    # Plot the Mel power-scaled frequency spectrum, with any factor of 128 frequency bins and 512 frames (frame default)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    print('mel', mel.shape)
    d = librosa.amplitude_to_db(mel)
    print('d', d.shape)
    librosa.display.specshow(d, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Power-Scaled Frequency Spectrogram')
    plt.tight_layout()
    plt.show()
    return mel


plotSpectrogram(y, sr)
