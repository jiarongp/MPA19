import numpy as np
import librosa
import IPython.display as ipd
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage


################################### Energy-Based ##########################################
def compute_novelty_energy(x, Fs=22050, N=1028, H=128, gamma=100, norm=1):
    """Compute energy-based novelty function 

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_energy: Energy-based novelty function 
        Fs_feature: Feature rate
    """    
    x_power = x**2
    w = signal.hann(N)
    # Frequency of novelty
    Fs_feature = Fs/H
    energy_local = np.convolve(x**2, w**2 , 'same')
    energy_local = energy_local[::H]
    if gamma!=None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    # because we need to pad the last element of the difference
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
    novelty_energy = np.copy(energy_local_diff)
    # half-rectification
    novelty_energy[energy_local_diff < 0] = 0
    if norm==1:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value
    return novelty_energy, Fs_feature

################################### Spectral-Based ##########################################

def compute_local_average(x, M, Fs=1):
    """Compute local average of signal

    Args:
        x: Signal
        M: Determines size (2M+1*Fs) of local average
        Fs: Sampling rate

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M * Fs))
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def compute_novelty_spectrum(x, Fs=22050, N=1024, H=256, gamma=100, M=10, norm=1):
    """Compute spectral-based novelty function

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    # difference of spectral componets along the axis of time
    Y_diff = np.diff(Y)
    # half-rectification
    Y_diff[Y_diff < 0] = 0
    # summing up differences for all frequencies
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

################################### Phase-Based ##########################################

def principal_argument(v):
    """Principal argument function 
    
    Args:
        v: value (or vector of values)
        
    Returns:
        w: Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w


def compute_novelty_phase(x, Fs=1, N=1024, H=64, M=40, norm=1):
    """Compute phase-based novelty function

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        M: Determines size (2M+1*Fs) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """     
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs/H
    phase = np.angle(X)/(2*np.pi)
    phase_diff = principal_argument(np.diff(phase, axis=1))
    phase_diff2 = principal_argument(np.diff(phase_diff, axis=1))
    novelty_phase = np.sum(np.abs(phase_diff2), axis=0)
    novelty_phase = np.concatenate( (novelty_phase, np.array([0, 0])) )    
    if M > 0:
        local_average = compute_local_average(novelty_phase, M)
        novelty_phase =  novelty_phase - local_average
        novelty_phase[novelty_phase<0]=0
    if norm==1: 
        max_value = np.max(novelty_phase)
        if max_value > 0:
            novelty_phase = novelty_phase / max_value
    return novelty_phase, Fs_feature

################################### Complex-domain novelty ##########################################

def compute_novelty_complex(x, Fs=1, N=1024, H=64, gamma=10, M=40, norm=1):
    """Compute complex-domain novelty function

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        M: Determines size (2M+1*Fs) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """     
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs/H
    mag = np.abs(X)
    if gamma > 0:
        mag = np.log(1 + gamma * mag)
    phase = np.angle(X)/(2*np.pi)
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.concatenate((phase_diff, np.zeros((phase.shape[0], 1))), axis=1)
    X_hat = mag*np.exp(2*np.pi*1j*(phase+phase_diff))
    X_prime = np.abs(X_hat - X)
    X_plus = np.copy(X_prime)
    for n in range(1, X.shape[0]):
        idx = np.where( mag[n,:] < mag[n-1,:] ) 
        X_plus[n, idx] = 0
    novelty_complex = np.sum(X_plus, axis=0)      
    if M > 0:
        local_average = compute_local_average(novelty_complex, M)
        novelty_complex =  novelty_complex - local_average
        novelty_complex[novelty_complex<0]=0
    if norm==1: 
        max_value = np.max(novelty_complex)
        if max_value > 0:
            novelty_complex = novelty_complex / max_value
    return novelty_complex, Fs_feature

################################### novelty curve resampling ############################################

def resample_signal(x_in, Fs_in, Fs_out=100, norm=1, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Args:
        x_in: Input signal
        Fs_in: Sampling rate of input signal 
        Fs_out: Sampling rate of output signal 
        norm: Apply max norm (if norm==1)
        time_max_sec: Duration of output signal (given in seconds)
        sigma: Standard deviation for smoothing Gaussian kernel

    Returns:
        x_out: Output signal
        F_out: Feature rate of output signal
    """    
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out     
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])    
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm==1:
        x_max = max(x_out) 
        if x_max > 0:
            x_out = x_out / max(x_out)    
    return x_out, Fs_out

################################### novelty comparison ############################################

def average_nov_dic(nov_dic, time_max_sec, Fs_out=100, norm=1, sigma=None):
    """Average respamples set of novelty functions

    Args:
        nov_dic: Dictionary of novelty functions
        time_max_sec: Duration of output signals (given in seconds)
        Fs_out: Sampling rate of output signal 
        norm: Apply max norm (if norm==1)
        sigma: Standard deviation for smoothing Gaussian kernel

    Returns:
        nov_matrix: Matrix containing resampled output signal (last one is average)
        Fs_out: Sampling rate of output signals 
    """       
    nov_num = len(nov_dic)
    N_out = int(np.ceil(time_max_sec*Fs_out))    
    nov_matrix = np.zeros([nov_num + 1, N_out])
    for k in range(nov_num):
        nov = nov_dic[k][0]
        Fs_nov = nov_dic[k][1]
        nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out, 
                                  time_max_sec=time_max_sec, sigma=sigma)      
        nov_matrix[k,:] = nov_out
    nov_average = np.sum(nov_matrix, axis=0)/nov_num
    if norm==1:
        max_value = np.max(nov_average)
        if max_value > 0:
            nov_average = nov_average / max_value        
    nov_matrix[k+1,:] = nov_average
    return nov_matrix, Fs_out 

################################### sonify ############################################

def sonify_noveltyCurve(novelty, x, Fs, sampling_frequency_novelty):
    # compare current novelty value with the previous one
    pos = np.append(novelty, novelty[-1]) > np.insert(novelty, 0, novelty[0])
    neg = np.logical_not(pos)
    peaks = np.where(np.logical_and(pos[:-1], neg[1:]))[0]
    
    values = novelty[peaks]
    values /= np.max(values)
    peaks = peaks[values >= 0.01]
    values = values[values >= 0.01]
    peaks_idx = np.int32(np.round(peaks / sampling_frequency_novelty * Fs))
    
    sine_periods = 8
    sine_freq = 880
    click = np.sin(np.linspace(0, sine_periods * 2 * np.pi, sine_periods * Fs//sine_freq))
    ramp = np.linspace(1, 1/len(click), len(click)) ** 2
    click = click * ramp
    click = (click * np.abs(np.max(x)))
    
    out = np.zeros(len(x), dtype=x.dtype)
    for i, start in enumerate(peaks_idx):
        if (start + len(click)) < len(x):
            idx = np.arange(start, start+len(click))
        else:
            idx = np.arange(start, len(x))
        out[idx] += (click[0:len(idx)] * values[i]).astype(x.dtype)
        
    ipd.display(ipd.Audio(np.vstack((x, out)), rate=Fs))

################################### HPSS ############################################

def horizontal_median_filter(B, filter_len):
    B_h = signal.medfilt(B, [1, filter_len])
    return B_h

def vertical_median_filter(B, filter_len):
    B_v = signal.medfilt(B, [filter_len, 1])
    return B_v


def HPSS(x, N, H, w='hann', Fs=22050, lh_sec=0.4, lp_Hz=1000):
    # x:      Input signal
    # N:      Frame length
    # H:      Hopsize
    # w:      Window function of length N
    # Fs:     Sampling rate of x
    # lh_sec: Horizontal median filter length given in seconds
    # lp_Hz:  Percussive median filter length given in Hertz

    # stft
    X = librosa.stft(x, n_fft=N, hop_length=H, window=w, center=True, pad_mode='constant')

    # power spectrogram
    Y = np.abs(X) ** 2

    # median filtering
    h = int(np.ceil(Fs*lh_sec/H))
    L_h = h + h%2 - 1
    p = int(np.ceil(N*lp_Hz/Fs))
    L_p = p + p%2 - 1 
    
    Y_h = horizontal_median_filter(Y, L_h)
    Y_p = vertical_median_filter(Y, L_p)

    # masking
    M_h = np.int8(Y_h >= Y_p)
    M_p = np.int8(Y_p > Y_h)
    
    X_h = X * M_h
    X_p = X * M_p

    # istft
    x_h = librosa.istft(X_h, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_p = librosa.istft(X_p, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    return x_h, x_p