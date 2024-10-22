#STANDARD BASICS IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# PLOT IMPORTS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

# SIGNAL IMPORTS
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

# WAVELET IMPORTS
import pywt

# CONSTANTS 
MAX_F = 40    # MAX Frequency for FFT plot
FS_UR5E = 500
MAX_FORCE_Z = 12
MAX_FORCE_Y = 12

# MAIN WINDOW CONSTANTS-FEATURES
WS = 1.6 # [s]
WS_B = int(WS * FS_UR5E) # WS = 1.6 --> 800

# 1ST TRANSIENT WINDOW, ENERGY BASED
ENERGY_WS = 0.6         # WINDOW DURATION [S] FOR THRESHOLD OVER THE WINDOW_SUM
STD_THRESHOLD = 0.82    # % used for MEAN - %*STD

# DISTANCE BEFORE FIRST FOUND TRANSIENT
DISTANCE_SECONDS = 0.314  # Desired distance for analysis in seconds
DISTANCE_SAMPLES = DISTANCE_SECONDS*FS_UR5E

def myfilter(signal, cutoff_freq=25, order=2, freq_sampling=500):
    #fc = 25, order = 4 or 6 too
    nyq_freq = 0.5 * freq_sampling
    normalized_cutoff = cutoff_freq / nyq_freq
    
    #Butterworth Low-Pass filter
    b, a = butter(order, normalized_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# CHECK IF SINGLE OR MULTIPLE TRANSIENT
def num_transient(signal, window_size=55, detail=3):
    #signal=np.abs(signal)
    threshold_half = max(signal, key=abs)/2
    threshold_std  = threshold_half-np.std(signal)*STD_THRESHOLD        # MEAN - %STD
    threshold_rstd = threshold_half-np.std(signal)                      # MEAN - STD
        
    newsignal_half = np.zeros_like(signal)
    newsignal_std  = np.zeros_like(signal)
    newsignal_rstd = np.zeros_like(signal)
    c_half, cnt_half = 0, 0
    c_std,  cnt_std  = 0, 0
    c_rstd, cnt_rstd = 0, 0
    
    for i in range (len(signal)):
        j = np.max(i-1, 0)
        
        #HALF
        if signal[i]>threshold_half : 
            newsignal_half[i]=1
            # if c_half==0: first_transient_half = i
            # c_half=1
            if newsignal_half[j] == 0: cnt_half+=1
            
        #STD
        if signal[i]>threshold_std : 
            newsignal_std[i]=1
            # if c_std==0: first_transient_std = i
            # c_std=1
            if newsignal_std[j] == 0: cnt_std+=1
        
        # REDUCED STD
        if signal[i]>threshold_rstd : 
            newsignal_rstd[i]=1
            # if c_rstd==0: first_transient_rstd = i
            # c_rstd=1
            if newsignal_rstd[j] == 0: cnt_rstd+=1
    
    # SHORT PERIODS OF 0 ARE PUTTED TO 1
    for i in range(len(newsignal_half)-window_size):
        window = newsignal_half[i:i+window_size]
        vals = np.sum(window)/window_size
        if all(window[0:detail]) == 1 and all(window[window_size-detail:window_size])==1:
            newsignal_half[i:i+window_size] = 1
        
        # STD
        window = newsignal_std[i:i+window_size]
        vals = np.sum(window)/window_size
        if all(window[0:detail]) == 1 and all(window[window_size-detail:window_size])==1:
            newsignal_std[i:i+window_size] = 1
        
        # RSTD
        window = newsignal_rstd[i:i+int(window_size*0.80)]
        vals = np.sum(window)/int(window_size*0.80)
        if all(window[0:detail]) == 1 and all(window[int(window_size*0.80)-detail:int(window_size*0.80)])==1:
            newsignal_rstd[i:i+int(window_size*0.80)] = 1
            
    # FIRST INDEX RECOUNTER AFTER HOMOGENEING
    cnt2_half, cnt2_std, cnt2_rstd = 0, 0, 0
    first_transient_half = []
    first_transient_std  = []
    first_transient_rstd = []
    
    for i in range(1, len(signal)):
        if newsignal_half[i]==1 and newsignal_half[i-1]==0: 
            first_transient_half.append(i)
            cnt2_half +=1
        if newsignal_std[i]==1  and newsignal_std[i-1]==0: 
            first_transient_std.append(i)
            cnt2_std  +=1
        if newsignal_rstd[i]==1 and newsignal_rstd[i-1]==0: 
            first_transient_rstd.append(i)
            cnt2_rstd +=1
    
    if cnt_half != cnt2_half: 
        #print(f'i have uniformed using window: cnt_half = {cnt_half} | cnt2_half = {cnt2_half}')
        cnt_half=cnt2_half
    
    if cnt_std != cnt2_std: 
        #print(f'i have uniformed using window: cnt_std = {cnt_std} | cnt2_std = {cnt2_std}')
        cnt_std=cnt2_std
    
    if cnt_rstd != cnt2_rstd: 
        #print(f'i have uniformed using window: cnt_rstd = {cnt_rstd} | cnt2_rstd = {cnt2_rstd}')
        cnt_rstd=cnt2_rstd
        
    return (first_transient_half, cnt2_half, newsignal_half, threshold_half), \
           (first_transient_std, cnt2_std, newsignal_std, threshold_std), \
           (first_transient_rstd, cnt2_rstd, newsignal_rstd, threshold_rstd)

def sliding_sum_window(signal, squared_signal_start, window_size=ENERGY_WS*FS_UR5E, accept_factor=0.25):
    len_sign = len(signal)
    max_sum = 0
    window_size = int(window_size)
    
    # find max (energy/window_sum)
    for i in range(len_sign-window_size):
        window = signal[i:i+window_size]
        summed = np.sum(window)
        if summed>max_sum: 
            max_sum=summed
            #max_sum_index=i
    min_sum_acepted = max_sum*accept_factor # % of the max summed values
    
    #print(squared_signal_start)
    for index in squared_signal_start:
        if np.sum(signal[index: min(index+window_size, len(signal))])>min_sum_acepted:
            return index
    
    return 0

def select_index(a, b, c, signal):
    vals = [a,b,c]
    vals = sorted(vals, reverse=True)
    a = vals[0]
    b = vals[1]
    c = vals[2]
    # SO A>B>C
    if np.abs(a - b) < 60 and np.abs(b - c) < 60:
        res = max(b - DISTANCE_SAMPLES, 0)
        if res > len(signal) - WS_B:    res = len(signal) - WS_B
        return int(res)
    else:
        res = max(c - DISTANCE_SAMPLES, 0)
        if res > len(signal) - WS_B:    res = len(signal) - WS_B
        return int(res)

def add_padding(signal, desired_length=WS_B):
    signal = np.array(signal)  # Convert signal to a NumPy array
    for i in range(len(signal)-1):
        if signal[i] is None and i > 1 and i < len(signal)-1 and signal[i-1] is not None and signal[i+1] is not None:
            signal[i] = (signal[i+1] + signal[i-1]) / 2
        elif signal[i] is None:
            signal[i] = np.mean(signal[max(i-10, 0):min(i+10, len(signal))])    #mean over the 20 prev and next
        
    if len(signal) < desired_length:
        padding_length = desired_length - len(signal)
        last_valid_value = signal[-1]  # Get the last valid value of the signal
        signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=last_valid_value)
        if len(signal) < desired_length:
            signal = np.append(signal, [signal[-1]] * (desired_length - len(signal)))  # Append last valid value

    return signal

def pad_signal_with_noise(signal, target_length=WS_B, noise_level=0.01):
    last_valid_value = signal[-1]
    padding_length = target_length - len(signal)
    padded_signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=last_valid_value)
    noise = np.random.normal(scale=noise_level, size=padding_length)
    padded_signal[-padding_length:] += noise
    return padded_signal

wavelets = ['morl', 'cmor', 'mexh', 'shan', 'cgau3', 'cgau5', 'gaus4', 'gaus5', 'fbsp']
scales = np.arange(0.25, 128)

def do_wavelet(signal, wavelet='morl'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return coefficients

def do_wavelets(signal, wavelets=wavelets, plot=False):
    coeffs_dict = {}
    if(plot):
        plt.plot(signal)
        plt.title("Sliced Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)  # Enable grid
        plt.show()
        
    if wavelets.shape == (1,):
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
        return coefficients
    else:
        # Perform CWT for each wavelet and store coefficients in the dictionary
        for wavelet in wavelets:
            coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
            coeffs_dict[wavelet] = coefficients
            if(plot):
                plt.figure(figsize=(8,4))
                plt.imshow(np.abs(coefficients)**2, aspect='auto', extent=[0, len(signal), 1, 128], cmap='jet')
                plt.colorbar(label='Magnitude')
                plt.gca().invert_yaxis()  # Invert y-axis
                plt.title(f"Scaleogram - {wavelet}")
                plt.xlabel('Time')
                plt.ylabel('Scale')
                plt.show()
            return coeffs_dict

def rename_and_convert_to_txt(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Split the file path into directory and filename
        directory, filename = os.path.split(file_path)
        
        # Add "DELETED" to the filename and change extension to .txt
        new_filename = "DELETED_" + os.path.splitext(filename)[0] + ".txt"
        
        # Create the new file path
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"File renamed to: {new_file_path}")
    else:
        print(f"File '{file_path}' does not exist.")