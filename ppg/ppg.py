import argparse
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut=0.5, highcut=4.0, order=10):
    """Bandpass filter the PPG signal to isolate heart rate range."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def load_wfdb_signal(filepath):
    """Load signal data from a WFDB (.dat/.hea) file."""
    record = wfdb.rdrecord(filepath)
    signal = record.p_signal[:, 0]  # Raw (unfiltered) signal
    sampling_rate = record.fs
    ft_signal = bandpass_filter(signal, sampling_rate)  # Filtered signal
    t = np.arange(len(signal)) / sampling_rate  # Time axis
    return t, signal, ft_signal, sampling_rate

def dft(x):
    """Compute the Discrete Fourier Transform (DFT) manually."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def plot_signal_fft(t, x, title="Signal", filename="signal_plot.png"):
    plt.figure(figsize=(12, 10))
    
    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    N = len(x)
    T = t[1] - t[0]
    xf = np.fft.fftfreq(N, T)[:N // 2]

    # FFT plot
    yf_fft = np.fft.fft(x)
    magnitude = 2.0 / N * np.abs(yf_fft[:N // 2])

    plt.subplot(2, 1, 2)
    plt.plot(xf, magnitude, label="FFT Magnitude")
    
    # Highlight the peak frequency
    peak_index = np.argmax(magnitude)
    peak_frequency = xf[peak_index]
    peak_magnitude = magnitude[peak_index]
    
    plt.plot(peak_frequency, peak_magnitude, 'ro')
    plt.text(peak_frequency, peak_magnitude, f'{peak_frequency:.2f} Hz', color='red',
            fontsize=12, ha='right', va='bottom')

    plt.title(f"{title} - Frequency Domain (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Signal plot saved to {filename}")
    plt.close()

    # Calculate heart rate
    heart_rate = peak_frequency * 60
    print(f"The highest frequency is: {peak_frequency:.2f} Hz")
    print(f"Heart rate: {heart_rate:.2f} beats per minute")

def plot_wavelet_comparison(t, x_raw, x_filtered, title="Wavelet Comparison", filename="wavelet_comparison.png"):
    scales = np.arange(1, 128)

    # Compute CWT for both signals
    coefficients_raw, _ = pywt.cwt(x_raw, scales, 'morl', sampling_period=t[1]-t[0])
    coefficients_filt, _ = pywt.cwt(x_filtered, scales, 'morl', sampling_period=t[1]-t[0])

    # Plot both scalograms
    plt.figure(figsize=(18, 6))

    # Unfiltered signal
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(coefficients_raw), extent=[t[0], t[-1], scales[-1], scales[0]],
               aspect='auto', cmap='jet', vmax=np.max(np.abs(coefficients_raw)))
    plt.colorbar(label='Magnitude')
    plt.title(f"{title} (Unfiltered)")
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")

    # Filtered signal
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(coefficients_filt), extent=[t[0], t[-1], scales[-1], scales[0]],
               aspect='auto', cmap='jet', vmax=np.max(np.abs(coefficients_filt)))
    plt.colorbar(label='Magnitude')
    plt.title(f"{title} (Filtered)")
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Wavelet comparison plot saved to {filename}")
    plt.close()

def plot_dft(t, x, title="DFT Transform", filename="dft_plot.png"):
    plt.figure(figsize=(12, 10))
    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    yf_dft = dft(x)
    N = len(x)
    T = t[1] - t[0]
    xf = np.fft.fftfreq(N, T)[:N // 2]

    plt.subplot(2, 1, 2)
    plt.plot(xf, 2.0 / N * np.abs(yf_dft[:N // 2]))
    plt.title(f"{title} - Frequency Domain (DFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(filename)
    print(f"DFT transform plot saved to {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Signal Analyzer for WFDB data")
    parser.add_argument('--file', type=str, required=True, help='Path to the .dat or .hea file')
    parser.add_argument('--dft', action='store_true', help='Include DFT')
    parser.add_argument('--wavelet', action='store_true', help='Include Wavelet Transform')

    args = parser.parse_args()

    if not os.path.exists(args.file + ".hea"):
        print(f"File not found: {args.file}")
        return

    # Load both raw and filtered signals
    t, x_raw, x_filtered, fs = load_wfdb_signal(args.file)
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    # Plot filtered signal and its FFT
    plot_signal_fft(t, x_filtered, title=f"{base_name} Filtered Signal", filename=f"{base_name}_plot.png")

    # Optionally plot the wavelet transform comparison
    if args.wavelet:
        plot_wavelet_comparison(t, x_raw, x_filtered, 
                               title=f"{base_name} Wavelet Comparison",
                               filename=f"{base_name}_wavelet_comparison.png")
    
    # Optionally plot the DFT transform
    if args.dft:
        if len(x_filtered) <= 1024:
            plot_dft(t, x_filtered, title="DFT Transform", filename=f"{base_name}_dft.png")
        else:
            print("Signal too long for DFT (N > 1024). Skipping DFT.")

if __name__ == "__main__":
    main()

