import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from butterworth import butter_bandpass_filter

# Settings
fs = 44100  # Sampling rate (samples per second)
duration = 3  # Duration of recording in seconds

# Record audio
print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Wait for the recording to finish
print("Recording complete.")

flatten_audio = audio.flatten()

# Time axis for plotting
t = np.linspace(0, duration, int(duration * fs), endpoint=False)

# Fourier Transform of original audio
fft_values = np.fft.fft(audio.flatten())  # Flatten to 1D
frequencies = np.fft.fftfreq(len(fft_values), 1/fs)

# Take the positive frequencies
half_n = len(frequencies) // 2
fft_magnitude = np.abs(fft_values[:half_n])
frequencies = frequencies[:half_n]

# Band-pass filter settings
lowcut = 50
highcut = 20_000

# Butterworth filter
filtered_audio = butter_bandpass_filter(flatten_audio, lowcut, highcut, fs, order=5)

# Fourier Transform of filtered audio
fft_values_filtered = np.fft.fft(filtered_audio)
frequencies_filtered = np.fft.fftfreq(len(fft_values_filtered), 1/fs)

# Take the positive frequencies for filtered signal
fft_magnitude_filtered = np.abs(fft_values_filtered[:half_n])
frequencies_filtered = frequencies_filtered[:half_n]

# Play original signal
print("Playing original audio...")
sd.play(audio, fs)
sd.wait()  # Wait until playback is done

# Play filtered signal
print("Playing filtered audio...")
sd.play(filtered_audio, fs)
sd.wait()  # Wait until playback is done

# Plotting
plt.figure(figsize=(12, 10))

# Plot original audio waveform (Time domain)
plt.subplot(2, 2, 1)
plt.plot(t, audio)
plt.title("Original Audio Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# Plot filtered audio waveform (Time domain)
plt.subplot(2, 2, 2)
plt.plot(t, filtered_audio)
plt.title("Filtered Audio Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# Plot original signal's Fourier Transform (Frequency domain)
plt.subplot(2, 2, 3)
plt.plot(frequencies, fft_magnitude)
plt.title("Original Audio Signal (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# Plot filtered signal's Fourier Transform (Frequency domain)
plt.subplot(2, 2, 4)
plt.plot(frequencies_filtered, fft_magnitude_filtered)
plt.title("Filtered Audio Signal (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
