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
        maxlev = pywt.dwt_max_level(len(ecg_data), pywt.Wavelet(wavelet).dec_length)
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


def fft_denoise(ecg_datas: np.ndarray, threshold=0.04):
    """使用FFT变换对ECG信号去噪

    Args:
        ecg_data (numpy.ndarray): 输入ECG数据,可以是1D或2D数组
        threshold (float): 阈值系数,默认0.04

    Returns:
        numpy.ndarray: 去噪后的ECG数据,与输入形状相同
    """
    new_ecg_data = []
    for ecg_data in ecg_datas:
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
        new_ecg_data.append(denoised_ecg.real)
    return np.array(new_ecg_data)


if __name__ == "__main__":
    pass
