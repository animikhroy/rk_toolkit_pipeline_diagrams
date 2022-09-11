#!/usr/bin/env python
import numpy as np
from scipy import signal
import logging

class Spectrogram():
    '''
    Builds a spectrogram from raw data
    '''
    def __init__(self, fs=1):
        self.fs = fs

    def from_raw_data(self, data):
        '''
        forwards to build spectrogram with appropriate args
        '''
        pass
   
    def build(self, event_time=1, time=1, deltat=5, freq_data=None):
        indxt = np.where((time >= event_time-deltat) & (time < event_time+deltat))
        logging.info("Index is {}".format(indxt))
        NFFT = int(self.fs/16.0)
        NOVL = int(NFFT*15./16)
        window = np.blackman(NFFT)
        f, t, Sxx = signal.spectrogram(x=freq_data[indxt], nfft=NFFT, noverlap=NOVL, fs=self.fs, window=window)
        return f,t,Sxx
