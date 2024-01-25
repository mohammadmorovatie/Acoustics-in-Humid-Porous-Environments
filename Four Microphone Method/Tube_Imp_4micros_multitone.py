#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:40:16 2023

@author: cecile
"""
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
from fonctions.Tube4Mic import Tube4Mic
from fonctions.measurement_DT9837 import measurement_DT9837

os.chdir(os.path.dirname(os.path.realpath(__file__)))
warnings.filterwarnings('ignore')

''' --------- Parameters --------- '''
fs = 48000                  # sample rate [Hz]
f1 = 50
f2 = 3000
T = 10
d = 23.5e-3                   # épaisseur d'échantillon [m]
PhPa = 1013
tc = 23
H = 50
S1 = 45e-3
S2 = 45e-3
l1 = 36.5e-3
l2 = 99.5e-3
fade = [0,0]


'''--------- Signal definition --------------'''
t4m = Tube4Mic(d, f1=f1, f2=f2, fs=fs, T=T, fade=fade, S1=S1, l1=l1, S2=S2, l2=l2, PhPa=PhPa, tc=tc, H=H)
# note that 't4tm' is an object.
x = t4m.multitone

''' --------- Measurement --------- '''
A = 1.5 #amplitude
temp_buffer = measurement_DT9837(A*x, fs)
in1 = np.array(temp_buffer[0::4])  # signal from input 0
in2 = np.array(temp_buffer[1::4])  # signal from input 1
in3 = np.array(temp_buffer[2::4])  # signal from input 2
in4 = np.array(temp_buffer[3::4])  # signal from input 3

x1 = in1[-(T+1)*fs:]  # élimination de la 1ere seconde (allumage)
x2 = in2[-(T+1)*fs:]
x3 = in3[-(T+1)*fs:]
x4 = in4[-(T+1)*fs:]



'''--------- Fonctions de transfert + fréquences --------------'''
f, uH21, uH31, uH41 = t4m.uHi1_mt(x1, x2, x3, x4) # uncorrected transfer functions H21, H31, H41 (H11=1)
npzfile = np.load('TubeImp4mic_calib_mt.npz') # calibration corrections
H21c, H31c, H41c = npzfile['H21c'], npzfile['H41c'], npzfile['H41c']
H21 = uH21/H21c
H31 = uH31/H31c
H41 = uH41/H41c
fig1, ax1 = plt.subplots(2,1)
ax1[0].set(title='Fonctions de transfert entre microphones')
ax1[0].plot(f, np.abs(H21),f, np.abs(H31),f, np.abs(H41))
ax1[0].set_ylabel('Amplitude')
ax1[0].set_xlim(200,2500)
ax1[0].legend(['H21', 'H31', 'H41'])
ax1[1].plot(f, np.angle(H21),f, np.angle(H31),f, np.angle(H41))
ax1[1].set(xlabel='Fréquence (Hz)', ylabel='Phase (rad)')
ax1[1].set_xlim(200,2500)


'''--------- Transfer matrix --------------'''
T11, T12_rc, T21rc = t4m.Tmatrix(f,H21,H31,H41)
fig2, ax2 = plt.subplots(6,1)
ax2[0].set(title='T11')
ax2[0].plot(f, np.abs(T11))
ax2[0].set_ylabel('Amplitude')
ax2[0].set_xlim(200,2500)
ax2[1].plot(f, np.angle(T11))
ax2[1].set_ylabel('Phase (rad)')
ax2[1].set_xlim(200,2500)
ax2[2].set(title='T12 / (rho*c)')
ax2[2].plot(f, np.abs(T12_rc))
ax2[2].set_ylabel('Amplitude')
ax2[2].set_xlim(200,2500)
ax2[3].plot(f, np.angle(T12_rc))
ax2[3].set_ylabel('Phase (rad)')
ax2[3].set_xlim(200,2500)
ax2[4].set(title='T21 * rho * c')
ax2[4].plot(f, np.abs(T21rc))
ax2[4].set_ylabel('Amplitude')
ax2[4].set_xlim(200,2500)
ax2[5].plot(f, np.angle(T21rc))
ax2[5].set(xlabel='Fréquence (Hz)', ylabel='Phase (rad)')
ax2[5].set_xlim(200,2500)


'''--------- Propriétés du matériau --------------'''
'''--------- Transmission et réflexion -------------'''
trans, TL = t4m.transmission(f, T11, T12_rc, T21rc) # transmission (fond anéchoïque)
R, alpha = t4m.reflexion(T11, T21rc) # réflexion sur fond rigide
fig3, ax3 = plt.subplots(2,1)
ax3[0].set(title='Propriétés de transmission et réflexion')
ax3[0].plot(f, TL)
ax3[0].set_ylabel('TL (dB)')
ax3[0].set_xlim(200,2500)
ax3[1].plot(f, alpha)
ax3[1].set(xlabel='Fréquence (Hz)', ylabel='alpha')
ax3[1].set_xlim(200,2500)


# '''--------- Nombre d'onde -------------'''
# K = t4m.wavenumber(T11)
# fig4, ax4 = plt.subplots(2,1)
# ax4[0].set(title='Nombre d''onde')
# ax4[0].plot(f, np.real(K))
# ax4[0].set_ylabel('Re(K) (1/m)')
# ax4[1].plot(f, np.imag(K))
# ax4[1].set(xlabel='Fréquence (Hz)', ylabel='Im(K) (1/m)')


'''--------- Impédance caractéristique -------------'''
Z = t4m.impedance(T12_rc, T21rc)
fig5, ax5 = plt.subplots(2,1)
ax5[0].set(title='Impédance caractéristique')
ax5[0].plot(f, np.real(Z))
ax5[0].set_ylabel('Re(Z) (Pa.s/m)')
ax5[0].set_xlim(200,2500)
ax5[1].semilogx(f, np.imag(Z))
ax5[1].set(xlabel='Fréquence (Hz)', ylabel='Im(Z) (Pa.s/m)')
ax5[1].set_xlim(200,2500)


'''------------ Sauvegarde des matrices de résultats ----------'''
""" SAVE the result to a numpy zip (.npz) file  """
np.savez_compressed('TubeImp4mic_mt.npz', f=f, H21=H21, H31=H31, H41=H41, T11=T11, T12_rc=T12_rc, T21rc=T21rc, trans=trans, R=R)
