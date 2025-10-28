import pywt
import numpy as np

from scipy.fft import fft, ifft


def wavelet_denoise(ecg_data: np.ndarray, threshold=0.04, wavelet="db8", mode="soft"):
    """使用小波变换对ECG信号去噪

    Args:
        ecg_data (numpy.ndarray): 输入ECG数据,可以是1D或2D数组
        threshold (float): 阈值系数,默认0.04
        wavelet (str): 使用的小波类型，默认'db8'
        mode (str): 阈值模式，'soft'或'hard'

    Returns:
        numpy.ndarray: 去噪后的ECG数据,与输入形状相同
    """

    if len(ecg_data.shape) == 1:
        datarec = []
        maxlev = pywt.dwt_max_level(len(ecg_data), pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec(ecg_data, wavelet, level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec.append(pywt.waverec(coeffs, wavelet))
        return np.array(datarec)
    elif len(ecg_data.shape) == 2:
        datarec = []
        for data in ecg_data:
            datarec.append(wavelet_denoise(data))
        return np.array(datarec)


def fft_denoise(ecg_data: np.ndarray, threshold=0.04):
    """使用FFT变换对ECG信号去噪

    Args:
        ecg_data (numpy.ndarray): 输入ECG数据,可以是1D或2D数组
        threshold (float): 阈值系数,默认0.04

    Returns:
        numpy.ndarray: 去噪后的ECG数据,与输入形状相同
    """
    if len(ecg_data.shape) == 1:
        # Apply FFT to the input data
        ecg_fft = fft(ecg_data)

        # Calculate the magnitude of the FFT coefficients
        magnitude = np.abs(ecg_fft)

        # Find the threshold for noise reduction
        cutoff = threshold * np.max(magnitude)

        # Set coefficients below the threshold to zero (remove noise)
        ecg_fft[magnitude < cutoff] = 0

        # Reconstruct the signal using inverse FFT
        denoised_ecg = ifft(ecg_fft)
    elif len(ecg_data.shape) == 2:
        denoised_ecg = []
        for data in ecg_data:
            denoised_signal = fft_denoise(data, threshold)
            denoised_ecg.append(denoised_signal)
    return np.array(denoised_ecg)


if __name__ == "__main__":
    # 示例用法
    import matplotlib.pyplot as plt

    # 生成示例ECG信号（带噪声）
    t = np.linspace(0, 1, 500)
    clean_ecg = np.sin(2 * np.pi * 5 * t)  # 示例干净信号
    noise = 0.5 * np.random.normal(size=t.shape)
    noisy_ecg = clean_ecg + noise

    # 使用小波去噪
    denoised_wavelet = wavelet_denoise(noisy_ecg, threshold=0.1)[0]

    # 使用FFT去噪
    denoised_fft = fft_denoise(noisy_ecg, threshold=0.1)[0]

    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Noisy ECG Signal")
    plt.plot(t, noisy_ecg)
    plt.subplot(3, 1, 2)
    plt.title("Wavelet Denoised ECG Signal")
    plt.plot(t, denoised_wavelet)
    plt.subplot(3, 1, 3)
    plt.title("FFT Denoised ECG Signal")
    plt.plot(t, denoised_fft)
    plt.tight_layout()
    plt.show()
