import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import scipy
import os
import math
import random
import pandas as pd
from tabulate import tabulate
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr

class AudioPreprocessor:
    @staticmethod
    def int_to_float(array, type=np.float32):
        """
        Change np.array int16 into np.float32
        Parameters
        ----------
        array: np.array
        type: np.float32
        Returns
        -------
        result : np.array
        """

        if array.dtype == type:
            return array

        if array.dtype not in [np.float16, np.float32, np.float64]:
            if np.max(np.abs(array)) == 0:
                array = array.astype(np.float32)
                array[:] = 0
            else:
                array = array.astype(np.float32) / np.max(np.abs(array))

        return array

    @staticmethod
    def float_to_int(array, type=np.int16, divide_max_abs=True):
        """
        Change np.array float32 / float64 into np.int16
        Parameters
        ----------
        array: np.array
        type: np.int16
        Returns
        -------
        result : np.array
        """

        if array.dtype == type:
            return array

        if array.dtype not in [np.int16, np.int32, np.int64]:
            if np.max(np.abs(array)) == 0:
                array[:] = 0
                array = type(array * np.iinfo(type).max)
            else:
                if divide_max_abs:
                    array = type(array / np.max(np.abs(array)) * np.iinfo(type).max)
                else:
                    array = type(array * np.iinfo(type).max)
        return array

    @staticmethod
    def remove_silence(y):
        threshold = 0.0005
        pause_length_in_ms = 800
        counter_below_threshold = 0
        indices_to_remove = []
        keep_at_start_and_end = 200
        
        for i, amp in enumerate(y):
            if abs(amp) < threshold:
                counter_below_threshold += 1
            else:
                if counter_below_threshold > pause_length_in_ms:
                    for index in range(i-counter_below_threshold+keep_at_start_and_end, i-keep_at_start_and_end):
                        indices_to_remove.append(index)
                counter_below_threshold = 0

        if counter_below_threshold > pause_length_in_ms:
            for index in range(len(y)-counter_below_threshold+keep_at_start_and_end, len(y)-keep_at_start_and_end):
                indices_to_remove.append(index)

        y_ = np.delete(y, indices_to_remove)

        return y_
    
    @staticmethod
    def clip(y, sr, seconds):
        return y[0:seconds*sr]
    
    @staticmethod
    def framing(y, sr, frame_size, overlap):
        frames = []
        if overlap >= frame_size or frame_size <= 0 or overlap < 0:
            return frames

        index = 0
        while index + frame_size < y.shape[0]:
            frames.append(y[index: index + frame_size])
            index = index + frame_size - overlap
        
        return frames
    
    @staticmethod
    def windowing(frames, window):
        windowed_frames = []
        for frame in frames:
            windowed_frames.append(frame * window(frame.shape[0]))
        return windowed_frames
      
class LPCCProcessor:    
    @staticmethod
    def lpc(frames, order=12):
        lpc_coefficients = []
        for frame in frames:
            lpc_coefficients.append(librosa.lpc(frame, order))
        return lpc_coefficients
    
    @staticmethod
    def lpcc(lpcs, order=12):
        lpcc_coefficients = []
        for lpc in lpcs:
            # lpc = lpc/np.max(np.abs(lpc))
            lpcc = np.zeros(order+1)
            lpcc[0] = lpc[0]
            lpcc[1] = lpc[1]
            for n in range (2, order+1):
                if n < lpc.shape[0] - 1:
                    lpcc[n] = sum((1-k/n) * lpc[k] * lpcc[n-k] for k in range(1, n)) + lpc[n]
                else:
                    lpcc[n] = sum((1-k/n) * lpc[k] * lpcc[n-k] for k in range(1, n))
            lpcc_coefficients.append(lpcc)
        return lpcc_coefficients
            
        
def plot(y, sr):
    fig, axs = plt.subplots(1)
    axs.plot(np.linspace(0, y.shape[0]/sr, y.shape[0]), y)
    
def writefile(y, sr, filename):
    sf.write(filename, y, sr)

def main():
    y, sr = librosa.load("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0002\\Speaker_0002_00000.wav")
    plot(y, sr)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    plot(y, sr)
    writefile(y, sr, "C:\\Users\\SCU8BH\\Documents\\T3000\\remove_noise.wav")
    y_ = AudioPreprocessor.remove_silence(y=y)
    y = y_ * np.max(np.abs(y))/np.max(np.abs(y_))
    plot(y, sr)
    writefile(y, sr, "C:\\Users\\SCU8BH\\Documents\\T3000\\remove_silence.wav")
    clip = AudioPreprocessor.clip(y, sr, 8)
    clip = clip/(np.max(np.abs(clip)))
    frames = AudioPreprocessor.framing(clip, sr, 1000, 100)
    windowed_frames = AudioPreprocessor.windowing(frames, np.hanning)
    lpcs = LPCCProcessor.lpc(windowed_frames)
    lpccs = LPCCProcessor.lpcc(lpcs)
    fig, axs = plt.subplots(4)
    for lpcc in lpccs:
        axs[0].plot(lpcc)
    print(lpcs[0])
    print(len(lpccs))
        
    y, sr = librosa.load("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0003\\Speaker_0003_00000.wav")
    lpccs = LPCCProcessor.lpcc(LPCCProcessor.lpc(AudioPreprocessor.windowing(AudioPreprocessor.framing(AudioPreprocessor.clip(AudioPreprocessor.remove_silence(nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)), sr, 8), sr, 1000, 100), np.hanning)))
    for lpcc in lpccs:
        axs[1].plot(lpcc)
    y, sr = librosa.load("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0004\\Speaker_0004_00000.wav")
    lpccs = LPCCProcessor.lpcc(LPCCProcessor.lpc(AudioPreprocessor.windowing(AudioPreprocessor.framing(AudioPreprocessor.clip(AudioPreprocessor.remove_silence(nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)), sr, 8), sr, 1000, 100), np.hanning)))
    for lpcc in lpccs:
        axs[2].plot(lpcc)
    y, sr = librosa.load("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0005\\Speaker_0005_00000.wav")
    lpccs = LPCCProcessor.lpcc(LPCCProcessor.lpc(AudioPreprocessor.windowing(AudioPreprocessor.framing(AudioPreprocessor.clip(AudioPreprocessor.remove_silence(nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)), sr, 8), sr, 1000, 100), np.hanning)))
    for lpcc in lpccs:
        axs[3].plot(lpcc)
    plt.show()

if __name__ == '__main__':
    main()