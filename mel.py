import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#	这里插入提取音频的路径
y, sr = librosa.load('dataset/UrbanSound8K/audio/swim/5.25 1 swim-g50-9.wav')

# 使用stft频谱求Mel频谱
D = np.abs(librosa.stft(y)) ** 2  # stft频谱
S = librosa.feature.melspectrogram(S=D)  # 使用stft频谱求Mel频谱

plt.figure(figsize=(8, 6))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
