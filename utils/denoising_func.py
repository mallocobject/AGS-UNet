import pywt
import numpy as np

from scipy.fft import fft, ifft


def wavelet_denoise(ecg_data, threshold=0.04):
    """using wavelet to denoise the ecg data

    Args:
        ecg_data (numpy.ndarray): the input ecg data, should be a 2d nparray or 3d nparray.

    Returns:
        np.array: the denoised ecg data, have the same shape as the input ecg data.
    """

    w = pywt.Wavelet("db8")
    if len(ecg_data.shape) == 2:
        datarec = []
        for data in ecg_data:

            maxlev = pywt.dwt_max_level(len(data), w.dec_len)
            coeffs = pywt.wavedec(data, "db8", level=maxlev)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec.append(pywt.waverec(coeffs, "db8"))
        return np.array(datarec)
    elif len(ecg_data.shape) == 3:
        datarec = []
        for data in ecg_data:
            datarec.append(wavelet_denoise(data))
        return np.array(datarec)


def fft_denoise(ecg_datas, threshold=0.04):
    """denoise via frequency fourier transform

    params:
      ecg_datas: the input ecg data, should be a 2D nparray or a list of 1D nparrays.
      alpha: meta-param for denoise, the threshold of the noise frequency which is the mean frequency. default to be 1.

    Output:
      denoised_data: a 2D ndarray which has the same shape as the input ecg data.
    """
    if isinstance(ecg_datas, list):
        ecg_datas = np.array(ecg_datas)

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
    # Example usage
    import matplotlib.pyplot as plt

    # Generate a sample ECG signal with noise
    t = np.linspace(0, 1, 500)
    clean_ecg = np.sin(2 * np.pi * 5 * t)  # Clean ECG-like signal
    noise = 0.5 * np.random.normal(size=t.shape)
    noisy_ecg = clean_ecg + noise

    # Denoise the signal using wavelet denoising
    denoised_ecg_wavelet = wavelet_denoise(np.array([noisy_ecg]), threshold=0.04)[0]

    # Denoise the signal using FFT denoising
    denoised_ecg_fft = fft_denoise(np.array([noisy_ecg]), threshold=0.04)[0]

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Noisy ECG Signal")
    plt.plot(t, noisy_ecg)
    plt.subplot(3, 1, 2)
    plt.title("Wavelet Denoised ECG Signal")
    plt.plot(t, denoised_ecg_wavelet)
    plt.subplot(3, 1, 3)
    plt.title("FFT Denoised ECG Signal")
    plt.plot(t, denoised_ecg_fft)
    plt.tight_layout()
    plt.show()
