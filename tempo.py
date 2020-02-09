import numpy as np
import matplotlib.pyplot as plt
import plot
import librosa
import IPython.display as ipd
from numba import jit
from scipy.interpolate import interp1d


################################### Fourier Tempogram ##########################################
@jit(nopython=True)
def compute_tempogram_Fourier(x, Fs, N, H, Theta=np.arange(30, 601, 1)):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Args:
        x: Input signal
        Fs: Sampling rate
        N: Window length
        H: Hop size
        Theta: Set of tempi (given in BPM)

    Returns:
        X: Tempogram
        T_coef: Time axis (seconds)
        F_coef_BPM: Tempo axis (BPM)
    """
    win = np.hanning(N)
    N_left = N // 2
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
#     x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(L_pad)
    # M is the time resolution
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
        T_coef = np.arange(M) * H / Fs
        F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM

################################### Autocorrelation Tempogram ##########################################

def compute_autocorrelation_local(x, Fs, N, H, norm_sum=True):
    """Compute local autocorrelation [FMP, Section 6.2.3]
    
    Args:
        x: Input signal
        Fs: Sampling rate
        N: Window length
        H: Hop size
        norm_sum: Normalizes by the number of summands in local autocorrelation

    Returns:
        A: Time-lag representation
        T_coef: Time axis (seconds)
        F_coef_lag: Lag axis
    """
    L = len(x)
    L_left = round(N / 2)
    L_right = L_left
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    if norm_sum is True:
        lag_summand_num = np.arange(N, 0, -1)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        if norm_sum is True:
            r_xx = r_xx / lag_summand_num
        A[:, n] = r_xx
    Fs_A = Fs / H
    T_coef = np.arange(A.shape[1]) / Fs_A
    F_coef_lag = np.arange(N) / Fs
    return A, T_coef, F_coef_lag

def compute_tempogram_autocorr(x, Fs, N, H, norm_sum=True, Theta=np.arange(30, 601)):
    """Compute autocorrelation-based tempogram

    Args:
        x: Input signal
        Fs: Sampling rate
        N: Window length
        H: Hop size
        norm_sum:
        Theta: Set of tempi (given in BPM)

    Returns:
        tempogram: Tempogram
        T_coef: Time axis (seconds)
        F_coef_BPM: Tempo axis (BPM)
        A_cut: Time-lag representation (cut according to Theta)
        F_coef_lag_cut: Lag axis
    """
    tempo_min = Theta[0]
    tempo_max = Theta[-1]
    lag_min = int(np.ceil(Fs * 60 / tempo_max))
    lag_max = int(np.ceil(Fs * 60 / tempo_min))
    A, T_coef, F_coef_lag = compute_autocorrelation_local(x, Fs, N, H, norm_sum=False)
    A_cut = A[lag_min:lag_max+1, :]
    F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
    F_coef_BPM_cut = 60 / F_coef_lag_cut
    F_coef_BPM = Theta
    tempogram = interp1d(F_coef_BPM_cut, A_cut, kind='linear',
                         axis=0, fill_value='extrapolate')(F_coef_BPM)
    return tempogram, T_coef, F_coef_BPM, A_cut, F_coef_lag_cut

################################### Cyclic Tempogram ##########################################

def compute_cyclic_tempogram(tempogram, F_coef_BPM, tempo_ref = 30, 
                             octave_bin = 40, octave_num = 4):
    """Compute cyclic tempogram

    Args:
        tempogram: Input tempogram
        F_coef_BPM: Tempo axis (BPM)
        tempo_ref: Reference tempo (BPM)
        octave_bin: Number of bin per tempo octave
        octave_num: Number of tempo octaves to be considered

    Returns:
        tempogram_cyclic: Cyclic tempogram
        F_coef_scale: Tempo axis with regard to scaling parameter
        tempogram_log: Tempogram with logarithmic tempo axis 
        F_coef_BPM_log: Logarithmic tempo axis (BPM)
    """    
    F_coef_BPM_log = tempo_ref * np.power(2, np.arange(0, octave_num*octave_bin)/octave_bin)
    F_coef_scale = np.power(2, np.arange(0, octave_bin)/octave_bin)
    tempogram_log = interp1d(F_coef_BPM, tempogram, kind='linear', axis=0, fill_value='extrapolate')(F_coef_BPM_log)
    K = len(F_coef_BPM_log)
    tempogram_cyclic = np.zeros((octave_bin, tempogram.shape[1]))
    for m in np.arange(octave_bin):
        tempogram_cyclic[m,:] = np.mean(tempogram_log[m:K:octave_bin,:], axis=0 )        
    return tempogram_cyclic, F_coef_scale, tempogram_log, F_coef_BPM_log

def set_yticks_tempogram_cyclic(ax, octave_bin, F_coef_scale, num_tick=5):
    """Set yticks with regard to scaling parmater

    Args:
        ax: Figure axis 
        octave_bin: Number of bin per tempo octave
        F_coef_scale: Tempo axis with regard to scaling parameter
        num_tick: Number of yticks
    """    
    yticks=np.arange(0,octave_bin,octave_bin//num_tick)
    ax.set_yticks(yticks)
    ax.set_yticklabels(F_coef_scale[yticks].astype((np.unicode_, 4)))
    
def plot_tempogram_Fourier_autocor(tempogram_F, tempogram_A, T_coef, F_coef_BPM, 
                                   octave_bin, title_F, title_A, norm=None):
    """Visualize Fourier-based and autocorrelation-based tempogram"""
    
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1,1]}, figsize=(12, 4))       

    output = compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
    tempogram_cyclic_F = output[0]
    F_coef_scale = output[1]
    if norm is not None:
        tempogram_cyclic_F = plot.normalize_feature_sequence(tempogram_cyclic_F, norm=norm)
    plot.plot_matrix(tempogram_cyclic_F, T_coef=T_coef, ax=[ax[0]], 
                         title=title_F, ylabel='Scaling', colorbar=True);
    set_yticks_tempogram_cyclic(ax[0], octave_bin, F_coef_scale, num_tick=5)

    output = compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
    tempogram_cyclic_A  = output[0]
    F_coef_scale = output[1]
    if norm is not None:
        tempogram_cyclic_A = plot.normalize_feature_sequence(tempogram_cyclic_A, norm=norm)    
    plot.plot_matrix(tempogram_cyclic_A, T_coef=T_coef, ax=[ax[1]], 
                         title=title_A, ylabel='Scaling', colorbar=True);
    set_yticks_tempogram_cyclic(ax[1], octave_bin, F_coef_scale, num_tick=5)

################################### Predominant Local Pulse ##########################################

@jit(nopython=True)
def compute_PLP(X, Fs, L, N, H, Theta=np.arange(30, 601)):
    """Compute windowed sinusoid with optimal phase

    Args:
        X: Fourier-based (complex-valued) tempogram
        Fs: Sampling rate
        N: Window length
        H: Hop size
        Theta: Set of tempi (given in BPM)

    Returns:
        nov_PLP: PLP function
    """
    
    win = np.hanning(N)
    N_left = N // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    nov_PLP = np.zeros(L_pad)
    M = X.shape[1]
    tempogram = np.abs(X)
    for n in range(M):
        # find out the maximum in time n
        k = np.argmax(tempogram[:, n])
        tempo = Theta[k]
        # compute the corresponding sinusoid frequency
        omega = (tempo / 60) / Fs
        c = X[k, n]
        phase = - np.angle(c) / (2 * np.pi)
        t_0 = n * H
        t_1 = t_0 + N
        # overlap and add
        t_kernel = np.arange(t_0, t_1)
        kernel = win * np.cos(2 * np.pi * (t_kernel * omega - phase))
        nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel
    nov_PLP = nov_PLP[L_left:L_pad-L_right]
    nov_PLP[nov_PLP < 0] = 0
    return nov_PLP

################################### Dynamic Programming Beat Tracking ##########################################

def compute_penalty(N, beat_ref):
    """Compute penalty funtion used for beat tracking [FMP, Section 6.3.2]
    Note: Concatenation of '0' because of Python indexing conventions

    Args:
        N: Length of vector representing penalty function
        beat_ref: Reference beat period (given in samples)

    Returns:
        penalty: Penalty function
    """    
    t = np.arange(1,N) / beat_ref
    penalty = -np.square(np.log2(t))
    t = np.concatenate((np.array([0]), t))
    penalty = np.concatenate((np.array([0]), penalty))
    return penalty

def compute_beat_sequence(novelty, beat_ref, penalty=None, factor=1, return_all=False):
    """Compute beat sequence using dynamic programming [FMP, Section 6.3.2]
    Note: Concatenation of '0' because of Python indexing conventions

    Args:
        novelty: Novelty function
        beat_ref: Reference beat period (given in samples)
        penalty: Penalty function (is computed when set to None)
        factor: Weight parameter for adjusting the penalty
        return_all: Return details (D, P)

    Returns:
        B: Optimal beat sequence
        D: Accumulated score  
        P: Maximization information 
    """      
    N = len(novelty)
    if penalty is None:
        penalty = compute_penalty(N, beat_ref)
    penalty = penalty * factor
    novelty = np.concatenate((np.array([0]), novelty))
    # D is accumulated scores
    D = np.zeros(N+1)
    P = np.zeros(N+1, dtype=int) 
    D[1] = novelty[1]
    P[1] = 0  
    #forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1,n)
        scores = D[m_indices] + penalty[n-m_indices]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1   
    #backtracking
    B = np.zeros(N, dtype=int)
    k = 0
    B[k] = np.argmax(D)
    while( P[B[k]]!=0 ):
        k = k+1
        B[k] = P[B[k-1]]
    B = B[0:k+1]
    B = B[::-1]
    B = B - 1
    if return_all:
        return B, D, P
    else:
        return B
    
def beat_period_to_tempo(beat, Fs):
    """Convert beat period (samples) to tempo (BPM) [FMP, Section 6.3.2]""" 
    tempo = 60 / (beat/Fs)
    return tempo
    
def compute_plot_sonify_beat(nov, Fs_nov, x, Fs, beat_ref, factor, title=None, figsize=(6,2)):
    """Compute, plot, and sonfy beat sequence from novelty function [FMP, Section 6.3.2]"""     
    B = compute_beat_sequence(nov, beat_ref=beat_ref, factor=factor)

    beats = np.zeros(len(nov))
    beats[np.array(B,dtype=np.int32)] = 1#
    if title is None:
        tempo = beat_period_to_tempo(beat_ref, Fs_nov)
        title = r'Optimal beat sequence ($\hat{\delta}=%d$, $F_\mathrm{s}=%d$, $\hat{\tau}=%0.0f$ BPM, $\lambda=%0.2f$)'%(beat_ref, Fs_nov, tempo, factor)

    fig, ax, line = plot.plot_signal(nov, Fs_nov, color='k', title=title, figsize=figsize)
    T_coef = np.arange(nov.shape[0]) / Fs_nov
    ax.plot(T_coef, beats, ':r', linewidth=1)
    plt.show()

    beats_sec = T_coef[B]
    x_peaks = librosa.clicks(beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))