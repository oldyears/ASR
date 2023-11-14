import numpy as np
import wave
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dct
from scipy.io import wavfile

# 定义PCA类
class PCA:
    # 初始化，设置目标维数、主成分和均值
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None  
        self.mean = None
    
    # 拟合方法，计算主成分、均值和协方差矩阵等信息
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)  #特征值和特征向量
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]

    # 转换方法，将输入数据转换为降维后的表示
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T) # 将数据投影到主成分上，得到降维后的数据表示
    
    # 结合拟合和转换两步骤，将输入数据转换为将为之后的表示
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# 调用plot函数画图
def DrawSpectogram(spec, name):
    fig = plt.figure(figsize=(25, 10))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Frames')
    # plt.tight_layout()
    plt.title(name)
    plt.savefig(name+'.jpg')
    plt.show()

# 预加重函数
def PreEmphasis(signal, coefficient = 0.97) :
    ''' 
    这里使用了numpy的小技巧避免循环，此代码使用错位取值来实现x(n)-a*x(n-1)
    '''
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

# 汉明窗 参考了np中对于特殊情况的处理
def MyHamming(N):
    if N < 1:
        return np.array([], dtype=np.result_type(N, 0.0))
    if N == 1:
        return np.ones(1, dtype=np.result_type(N, 0.0))
    n = np.arange(0, N)
    return 0.54 + 0.46*np.cos(2*np.pi*n/(N-1))
        
# 分帧和加窗
def AudioToFrame(signal, sample_rate, frame_size = 0.025, frame_stride = 0.01):
    signal_length = len(signal) # 计算信号的总长度
    frame_length = int(round(frame_size * sample_rate)) # 每帧的采样点数
    frame_step = int(round(frame_stride * sample_rate)) # 每帧的步长
    num_frames = int(np.ceil((signal_length - frame_length)/frame_step)) + 1    # 得到总帧数
    pad_signal_length = (num_frames - 1) * frame_step + frame_length    # 计算补0之后的信号长度
    pad_signal = np.append(signal, np.zeros(pad_signal_length - signal_length))
    indices_1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    indices_2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),(frame_length,1)).T
    indices = indices_1 + indices_2 # 计算每一帧的索引
    frames = pad_signal[indices]    
    window = MyHamming(frame_length) # 创建汉明窗
    # window = np.hamming(frame_length)
    return frames * window  # 对每帧的信号进行加窗处理

# SIFT即短时傅里叶变化
def STFT(signal):
    NFFT = 512  # 定义FFT的点数，一般为2的整次幂
    spectrum = np.fft.rfft(signal, NFFT)    # 对每帧的信号进行快速傅里叶变换
    return spectrum

# 频率转梅尔频率转换函数
def hz_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700.0)

# 梅尔频率转频率函数
def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

# Mel滤波器组函数
def MelFilterBank(sample_rate,num_filter = 26, NFFT = 512):
    low_freq = 0 # 最低频率
    high_freq = sample_rate / 2 # 定义最高频率
    mel_low = hz_to_mel(low_freq)   # 对频率进行梅尔频率转换
    mel_high = hz_to_mel(high_freq)

    # 得到梅尔频率后进行计算
    mel_points = np.linspace(mel_low, mel_high, num_filter + 2) # 在梅尔频率上均匀地取num_filter+2个点
    hz_points = mel_to_hz(mel_points)   
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)    # 计算每个频率对应的FFT bin号
    filter_bank = np.zeros((num_filter, int(NFFT / 2 + 1)))

    # 遍历每个滤波器并存储到矩阵中，根据公式计算即可
    for m in range(1, num_filter + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        # 计算增益
        for k in range(f_m_minus, f_m): 
            filter_bank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            filter_bank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return filter_bank

# DCT函数，直接调用函数即可
def DCT(signal, num_cepstra = 12):
    return dct(signal, type=2, axis=1, norm='ortho')[:,:num_cepstra]

# 定义动态特征提取函数
def Delta(feat, N):
    num_frames = len(feat)  # 计算总帧数
    denominator = 2 * sum([n ** 2 for n in range(1, N + 1)])    # 计算分母
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')    # 对特征进行边缘填充
    for t in range(num_frames): # 遍历每一帧，并计算差分特征
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t:t + 2 * N + 1]) / denominator
    return delta_feat

# 定义特征变换函数
def FeatureTransform(feat):
    # 计算每个维度的均值和标准差，以进行归一化处理
    mean = np.mean(feat, axis=0)
    std = np.std(feat, axis=0)
    feat = (feat - mean) / std

    # 对特征进行PCA降维处理
    pca = PCA(n_components=10)
    feat = pca.fit_transform(feat)
    return feat



# 数据准备
sample_rate, signal = wavfile.read('test_record.wav')

# 进行预加重操作
pre_emphasised_signal = PreEmphasis(signal)

# 分帧加窗
windowed_signal = AudioToFrame(pre_emphasised_signal, sample_rate)

# STFT
stfted_signal = STFT(windowed_signal)

# Mel滤波
signal_2 = np.abs(stfted_signal) ** 2
filter_bank = MelFilterBank(sample_rate)
filtered_signal = np.dot(signal_2, filter_bank.T)

# 取对数
loged_signal = np.log(filtered_signal)

# DCT处理
dcted_signal = DCT(loged_signal)

# 动态差分处理
delta_1 = Delta(dcted_signal, 2)
delta_2 = Delta(delta_1, 2)
deltaed_signal = np.concatenate((dcted_signal, delta_1, delta_2), axis=1)

# 特征提取
feature_signal = FeatureTransform(deltaed_signal)

# 给出最后的MFCC图谱
DrawSpectogram(feature_signal.T,'final feature')





