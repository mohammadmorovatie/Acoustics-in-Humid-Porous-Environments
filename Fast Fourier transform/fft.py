#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:04:00 2024

@author: mohammad
"""
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Load data from the first file 'C3__Tube_eau_sweep.dat'
data_c3 = np.loadtxt('C3__Tube_eau_sweep.dat')
t_c3 = data_c3[:, 0]  # Extract time values from the first column
amp_c3 = data_c3[:, 1]  # Extract amplitude values from the second column

# Load data from the second file 'C2__Tube_eau_sweep.dat'
data_c2 = np.loadtxt('C2__Tube_eau_sweep.dat')
t_c2 = data_c2[:, 0]  # Extract time values from the first column
amp_c2 = data_c2[:, 1]  # Extract amplitude values from the second column

# Parameters for the Fourier transform
n = 2**15  # Number of points for the Fourier transform
df = 1 / (t_c3[4] - t_c3[3]) / n  # Frequency resolution
f = np.arange(1, n + 1) * df  # Frequency vector based on the resolution

# Compute the Fourier transform for the first file
sft_c3 = np.fft.fft(amp_c3, n)

# Compute the Fourier transform for the second file
sft_c2 = np.fft.fft(amp_c2, n)

# Plotting for the first file
plt.figure()
plt.plot(t_c3, amp_c3, label='C3__Tube_eau_sweep.dat', color='blue')  # Plot time domain, specify color, and add label for legend
plt.xlabel('Time (s)')  # Label for the x-axis
plt.ylabel('Amplitude')  # Label for the y-axis
plt.legend()  # Display legend

plt.figure()
plt.plot(f, np.abs(sft_c3), label='C3__Tube_eau_sweep.dat', color='blue')  # Plot frequency spectrum, specify color, and add label for legend
plt.xlabel('Frequency (Hz)')  # Label for the x-axis
plt.ylabel('Amplitude')  # Label for the y-axis
plt.xlim([1, 2e3])  # Set x-axis limits for better visibility
plt.legend()  # Display legend

# Plotting for the second file
plt.figure()
plt.plot(t_c2, amp_c2, label='C2__Tube_eau_sweep.dat', color='green')  # Plot time domain, specify color, and add label for legend
plt.xlabel('Time (s)')  # Label for the x-axis
plt.ylabel('Amplitude')  # Label for the y-axis
plt.legend()  # Display legend

plt.figure()
plt.plot(f, np.abs(sft_c2), label='C2__Tube_eau_sweep.dat', color='green')  # Plot frequency spectrum, specify color, and add label for legend
plt.xlabel('Frequency (Hz)')  # Label for the x-axis
plt.ylabel('Amplitude')  # Label for the y-axis
plt.xlim([1, 2e3])  # Set x-axis limits for better visibility
plt.legend()  # Display legend

plt.show()  # Show the plots
