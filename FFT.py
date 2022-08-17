from __future__ import division, print_function
from numpy.fft import rfft, ifft
from numpy import array, int16, arange, linspace, abs, where, allclose, empty, zeros
from scipy.io.wavfile import read, write
from pylab import plot, show, xlabel, ylabel, subplots, grid, xlim, ylim, title, setp, axes
import pandas as pd
import scipy
from scipy import fftpack

# Read in .wav file and put in array called data. (Taken from the provided .py program)
inwav = read("hidden1.wav")   # Read in sound sample.
samprate = inwav[0]         # Extract sample rate.
data = inwav[1][:, 0]       # Extract data list.
amplitude = array(data)


# Plotting Amplitude vs time original
x = linspace(0, (len(amplitude) / samprate), len(amplitude))

# print(len(amplitude))
# print(samprate)
# print(x)
# print(x.shape)
# print(amplitude.shape)
# xlabel('time(s)->')
# ylabel('<-amplitude->')
# plot(x, amplitude)
# show()

# transforming amplitudes to frequency and plotting freq vs amp
fft_signal = fftpack.fft(amplitude)
power = abs(fft_signal)
sample_freq = fftpack.fftfreq(amplitude.size, d=1 / samprate)
# plot(sample_freq, power)
# xlim(-6000, 6000)
# xlabel('Frequency(Hz)')
# ylabel('Power')
# title('Max Frequencies')
# show()

# finding peak frequency
pos_mask = where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
# xlabel('Frequency(Hz)')
# ylabel('Magnitude')
# title('Figure showing Peak Frequency')
# plot(freqs[:8], power[:8])
# show()

# filtering frequncies with amp higher than 10000
high_freq = fft_signal.copy()
new_freq = zeros(len(high_freq))
print(len(new_freq))
for i in range(len(high_freq)):
    if(high_freq[i] < 1000):
        new_freq[i] = high_freq[i]
    else:
        new_freq[i] = 0


# plotting the new amp vs time using the new frequency
filtered = fftpack.ifft(new_freq)
title('After removing higher frequencies, secret message: Physics 239')
ylim(-20000, 20000)
xlabel('time(s)->')
ylabel('<-amplitude->')
plot(x, filtered)
show()

# Normalize and output sound clip.
maxdata = 32767 / max(abs(filtered))
normdata = maxdata * filtered
data2 = array([normdata, normdata], dtype=int16).T
# Save as .wav file.
write("After_removing_freq.wav", samprate, data2)


# Plotting Amplitude vs time original
# x = linspace(0, (len(amplitude) / samprate), len(amplitude))

# print(len(amplitude))
# print(samprate)
# print(x)
# print(x.shape)
# print(amplitude.shape)
# xlabel('time(s)->')
# ylabel('<-amplitude->')
# plot(x, amplitude)
# show()
