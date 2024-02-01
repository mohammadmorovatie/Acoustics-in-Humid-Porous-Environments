#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:03:07 2024

@author: mohammad
"""
import warnings  # Importing the warnings module to handle warnings
import os  # Importing the os module for operating system related functions
import time  # Importing the time module for time-related functions
import numpy as np  # Importing NumPy library for numerical computations
import h5py  # Importing h5py library for working with HDF5 files
from numpy.fft import rfft, rfftfreq  # Importing FFT functions from NumPy
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from functions.measurement_DT9837 import measurement_DT9837  # Importing a custom measurement function

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # Setting the current directory to the script's directory
warnings.filterwarnings('ignore')  # Suppressing warnings

start = time.time()  # Recording the start time for performance measurement

''' --------- Measure conditions --------- '''
T, rho = 23.4, 1.2  # Temperature and density
c = 20.05 * np.sqrt(273.15 + T)  # Speed of sound calculation

''' --------- Parameters --------- '''
fs = 48000  # Sampling frequency
A = 1.5  # Amplitude
S1 = S2 = 0.045  # Dimensions
f_l, f_u = 400, 2500  # Lower and upper frequency bounds
d = 0  # sample thickness
x1, x2 = 0.0815, 0.0365  # Microphones Positions

name_sample = "Air_Impdance_tube"  # Sample name
num = "1"  # Sample number

''' --------- Signal Definition --------- '''
T = 2  # Duration of the signal
t = np.arange(0, T, 1/fs)  # Time vector

freq = np.unique(np.round(np.logspace(np.log10(f_l), np.log10(f_u), 800)))  # Frequency vector
x = 0  # Initializing the signal
for f0 in freq:  # Looping over frequencies
    x += np.sin(2*np.pi*f0*t + 2*np.pi*np.random.rand())  # Generating sinusoidal signal with random phase

x /= np.max(np.abs(x))  # Normalizing the signal

''' --------- Measurements --------- '''
temp_buffer = measurement_DT9837(A*x, fs)  # Acquiring signals from measurement device
voie0 = np.array(temp_buffer[0::4])[-fs:]  # Extracting signal from input 0
voie1 = np.array(temp_buffer[1::4])[-fs:]  # Extracting signal from input 1

voie0 = voie0 - np.mean(voie0)  # Removing DC offset
voie1 = voie1 - np.mean(voie1)  # Removing DC offset

X0, X1 = rfft(voie0), rfft(voie1)  # Calculating FFT of signals
P0, P1 = X0[freq.astype('int')], X1[freq.astype('int')]  # Power spectrum of signals

freq_axis = rfftfreq(len(voie0), 1/fs)  # Frequency axis
freq_axis = [freq_axis.astype('int')]  # Converting frequency axis to integers
t_axis = t[-fs:]  # Time axis
k = (2*np.pi*freq) / c  # Wavenumber calculation

''' --------- Tranfert function --------- '''
H10 = P1 / P0  # Transfer function between signal 1 and signal 0

H_I = np.exp(-1j * k * (x1 - x2))  # Incident wave
H_R = np.exp(1j * k * (x1 - x2))  # Reflected wave

R = (H10 - H_I) / (H_R - H10) * np.exp(2 * 1j * k * x1)  # Reflection coefficient
Alpha = 1 - np.abs(R)**2  # Absorption coefficient


''' --------- Plotting --------- '''
plt.figure(figsize=(16, 5))  # Creating a new figure for plotting
plt.subplot(121)  # Creating subplot for absorption coefficient
plt.plot(freq, Alpha, 'k', linewidth=3)  # Plotting absorption coefficient
plt.ylabel(r'$\alpha$', fontsize=20)  # Labeling y-axis
plt.xlim(f_l, f_u)  # Setting x-axis limits
plt.ylim(0, 1)  # Setting y-axis limits
plt.grid()  # Adding grid
plt.xticks(fontsize=18)  # Setting x-axis tick font size
plt.yticks(fontsize=18)  # Setting y-axis tick font size
plt.xlabel('Fréquences [Hz]', fontsize=20)  # Labeling x-axis
plt.title('Absorption Coefficient', fontsize=20)  # Adding title
plt.tight_layout()  # Adjusting layout

# Plotting real part of reflection coefficient
plt.subplot(122)  # Creating subplot for reflection coefficient
plt.plot(freq, np.real(R), 'b', label='Real Part', linewidth=3)  # Plotting real part
plt.plot(freq, np.imag(R), 'r', label='Imaginary Part', linewidth=3)  # Plotting imaginary part
plt.ylabel('Reflection Coefficient', fontsize=20)  # Labeling y-axis
plt.xlim(f_l, f_u)  # Setting x-axis limits
plt.grid()  # Adding grid
plt.xticks(fontsize=18)  # Setting x-axis tick font size
plt.yticks(fontsize=18)  # Setting y-axis tick font size
plt.xlabel('Fréquences [Hz]', fontsize=20)  # Labeling x-axis
plt.title('Reflection Coefficient', fontsize=20)  # Adding title
plt.legend(fontsize=16)  # Adding legend
plt.tight_layout()  # Adjusting layout

plt.show()  # Displaying the plot

''' ---------Saving Data --------- '''

# Creating an HDF5 file with the sample name and number
filename = f"{name_sample}_data_{num}.h5"  # Define filename based on sample name and number
with h5py.File(filename, 'w') as hf:  # Open the HDF5 file in write mode
    # Saving parameters
    hf.attrs['Temperature'] = T  # Saving temperature attribute
    hf.attrs['Density'] = rho  # Saving density attribute
    hf.attrs['Speed_of_sound'] = c  # Saving speed of sound attribute
    hf.attrs['Sampling_frequency'] = fs  # Saving sampling frequency attribute
    hf.attrs['Amplitude'] = A  # Saving amplitude attribute
    hf.attrs['Dimensions'] = [S1, S2]  # Saving dimensions attribute
    hf.attrs['Frequency_range'] = [f_l, f_u]  # Saving frequency range attribute
    hf.attrs['Distances'] = [d, x1, x2]  # Saving distances attribute
    hf.attrs['Sample_name'] = name_sample  # Saving sample name attribute
    hf.attrs['Sample_number'] = num  # Saving sample number attribute
    
    # Saving calculated data
    hf.create_dataset('Frequency', data=freq)  # Saving frequency data
    hf.create_dataset('Absorption_coefficient', data=Alpha)  # Saving absorption coefficient data
    hf.create_dataset('Real_part_of_reflection_coefficient', data=np.real(R))  # Saving real part of reflection coefficient data
    hf.create_dataset('Imaginary_part_of_reflection_coefficient', data=np.imag(R))  # Saving imaginary part of reflection coefficient data
    
    
    
print("Data saved to", filename)  # Print confirmation message with filename

end = time.time()
temps = end - start
print("Total Elapsed Time= {} secondes".format(np.round(temps, 1)))