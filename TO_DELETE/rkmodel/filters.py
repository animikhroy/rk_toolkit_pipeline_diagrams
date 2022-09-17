#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz, spectrogram
from scipy import signal

def whiten_data(strain_data, time, fs, fband):
    logger.info("Whitening data..")
    dt = time[1] - time[0]
    NFFT = 4*data['fs']
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    for x in ['H1', 'L1']:
        pxx, freqs = mlab.psd(strain_data, Fs = fs, NFFT = NFFT)
        psd_H1 = interp1d(freqs, pxx)
        strain_data = whiten(strain_data, psd_H1, dt) # uncleaf if this is required
        return filtfilt(bb, ab, strain_data) / normalization
    return data

# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0,2048.,int(Nt/2+1))
    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

