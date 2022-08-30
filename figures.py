import matplotlib as mpl
# print(mpl.rcParams.keys())
# # input()
new_rc_params = {#'text.usetex': True,
         # 'svg.fonttype': 'none',
         # 'font.sans-serif': 'Helvetica',
         # 'text.latex.preamble': r'\usepackage{libertine}',
         'font.size': 7,
         'font.family': 'Montserrat',
         # 'mathtext.fontset': 'custom',
         # 'mathtext.rm': 'Helvetica Neue',
         # 'mathtext.it': 'Helvetica Neue:italic',
         # 'mathtext.bf': 'Helvetica Neue:bold',
         'axes.linewidth': 0.1,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7,
         'hatch.linewidth': 0.01,
         'legend.fontsize':7,
         'legend.handlelength': 2
         }

mpl.rcParams.update(new_rc_params)


import librosa
import librosa.display
import matplotlib.pyplot as plt
# plt.rc('font', family='Helvetica')
import pandas as pd
import seaborn as sns
import os
import numpy as np
import glob
# import json
from json import load
import slice_audio
from sklearn.preprocessing import normalize
import stream_latency

def set_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX. Taken from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    # width = 455.24411
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

""" Plot Example Keystroke, Showing Peaks """
def waveplot_with_peaks(fname, peaks, duration=0.150, savefig=True):
    width = 232.74377
    # width=345
    y, _ = librosa.load(fname, duration=duration+0.001)

    plt.figure(figsize=set_size(width, fraction=1))
    librosa.display.waveplot(y, x_axis='ms', linewidth=.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)

    i = 0
    for k, v in peaks.items():
        plt.vlines(v[0], ymax=v[1], ymin=-v[1], linestyle='--', color='r', alpha=0.5, linewidth=0.5, label='Peaks')
        if i == 0:
            # plt.legend()
            pass

        plt.text(x=v[0], y=-v[1] - .2, s=k, ha='center', fontsize=5)
        i += 1


    if savefig:
        fname, _ = os.path.splitext(os.path.split(fname)[1])
        fpath = os.path.join('presentation_figures', fname)
        plt.savefig(fpath, bbox_inches='tight', dpi=600)


#""" Determine Maximum TDOA For A Given Delta_M and a Given A_MAX """
#def theoretical_tdoa(delta_m, Fs, label):
#    v_sound = 343
#    tau = [] 
#    A = []
#    for A_max in np.arange(0, 0.5, 0.001):
#        A.append(A_max)
#        tau.append(((A_max**2 + delta_m**2)**0.5 - A_max) * (Fs / v_sound))
#    plt.plot(A, tau, label=label)
#    return tau
#    # plt.xlim(0, 0.5)
#    # plt.ylim(0, 80)

#width = 232.74377
#plt.figure(figsize=set_size(width, fraction=1))
#plt.xlim(0, 0.5)
#plt.ylim(0, 120)
#plt.xlabel('$A_{max}$ [m]')
#plt.ylabel('Maximum TDoA [samples]')
#tau1 = theoretical_tdoa(0.279, 96000, 'Apple A1644')
#tau2 = theoretical_tdoa(0.380, 96000, 'Logitech K780')
#plt.vlines(0.15, 0, tau2[150], linestyle='--', color='r', alpha=.2)
#plt.hlines(tau1[150], 0, 0.15, linestyle='--', color='r', alpha=.2)
#plt.hlines(tau2[150], 0, 0.15, linestyle='--', color='r', alpha=.2)
#plt.legend()
#fpath = os.path.join('dissertation_article', 'res', 'img', 'theoretical_tdoa')
#plt.savefig(fpath, bbox_inches='tight', dpi=1200)
##plt.show()

f = os.path.join('dissertation_article', 'res', 'audio', 'example_keystroke.wav')
peaks = {'Touch': (0.0095, .20), 'Hit': (0.0160, .60), 'Release': (0.1250, .20)}
waveplot_with_peaks(f, peaks)
plt.show()

#""" Plot Unix Times Against Example Keystrokes """
#def plot_unix_times(fname, fpath_json, savefig=True):
#    width = 232.74377
#    y, sr = librosa.load(fname, sr=None)
#    # x = 0.255

#    plt.figure(figsize=set_size(width, fraction=1))
#    # plt.figure(figsize=set_size(width))
#    librosa.display.waveplot(y, sr=sr, x_axis='s')
#    plt.xlabel('Time (s)')
#    plt.ylabel('Amplitude')
#    plt.ylim(-1, 1)

#    with open(fpath_json) as f:
#        data = slice_audio.DotDict(load(f))
#        events = data.events

#    i = 0
#    for event in events:
#        event = slice_audio.DotDict(event)
#        time_s = (event.delta_time / 1E+09) - 0.854
#        if event.event_type == 'KEY_DOWN':
#            plt.vlines(time_s, ymin=-0.75, ymax=0.75, color='r', alpha=0.5, linestyle='--', label='Press', linewidth=0.5)
#            if i == 0:
#                plt.legend()
#        else:
#            plt.vlines(time_s, ymin=-0.75, ymax=0.75, color='g', alpha=0.5, linestyle='--', label='Release', linewidth=0.5)
#            if i == 1:
#                plt.legend()
#        i += 1

#    if savefig:
#        fname, _ = os.path.splitext(os.path.split(fname)[1])
#        fpath = os.path.join('dissertation_article', 'res', 'img', fname)
#        plt.savefig(fpath, bbox_inches='tight', dpi=1200, pad_inches=0.0)

## f = os.path.join('dissertation', 'res', 'audio', 'unix_time.wav')
## json = os.path.join('dissertation', 'res', 'unix.json')

### f = os.path.join('code', 'src', 'data', 'audio', 'SAMPLE_TEXT-USER02.wav')
### json = os.path.join('code', 'src', 'data', 'SAMPLE_TEXT-USER02.json')

### PLOT WAVEPLOT:
## plot_unix_times(f, json)
## plt.savefig('dissertation_article/res/img/audio_time_series')
##plt.show()

##""" Plot Onset Strength"""
### f = os.path.join('dissertation_article', 'res', 'audio', 'unix_time.wav')
##width = 232.74377/1.5
##_, h = set_size(width)
##width = 489.38776
##w, _ = set_size(width, fraction=1)
### w = w*2
### f = os.path.join('code', 'src', 'data', 'audio', 'USER02', 'SAMPLE_TEXT', 'STEREO','CH03', 'a', 'a1.wav')
### plt.figure(figsize=(w, h))
### y, sr = librosa.load(f, sr=None, offset=2.625, duration=1.51)
### D = librosa.stft(y)
### o_env = librosa.onset.onset_strength(y, sr=sr)
### times = librosa.times_like(o_env, sr=sr)
### onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, pre_max=1, post_max=1, pre_avg=50, post_avg=50, delta=0.4, wait=10)
### times = librosa.frames_to_time(np.arange(D.shape[1]), sr)
### onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
### plt.plot(times, onset_env / onset_env.max(), label='Median (custom mel)', linewidth=.5)
### plt.vlines(times[onset_frames], 0, 1, color='r', alpha=0.5, linestyle='--', label='Onsets', linewidth=.5)
### plt.xlabel('Time (s)')
### plt.ylabel('Normalised Strength')
### plt.yticks(np.arange(0, 1.01, 0.5))
### plt.tight_layout()
### plt.savefig('dissertation_article/res/img/spectral_flux', dpi=1200, bbox_inches='tight', pad_inches=0.0)
### #plt.show()


##width = 232.74377
##unrefined = os.path.join('dissertation_article', 'res', 'audio', 'keystroke_unrefined.wav')
##refined = os.path.join('dissertation_article', 'res', 'audio', 'keystroke_refined.wav')
##audio = slice_audio.AudioUtil(unrefined)
##peak = stream_latency.get_onset_times(audio.y,audio.sr)
##plt.figure(figsize=set_size(width))
##librosa.display.waveplot(audio.y, sr=audio.sr, x_axis='ms', linewidth=.5)
##plt.ylabel('Amplitude')
##plt.vlines(peak-0.015, 1, -1,  color='r', linestyle='--', alpha=.5, linewidth=.5)
##plt.vlines(peak+0.135, 1, -1, color='r', linestyle='--', alpha=.5, linewidth=.5)
##plt.savefig('dissertation_article/res/img/refined_keystroke', dpi=1200, bbox_inches='tight', pad_inches=0.0)
###plt.show()





### #plt.show()

###""" Plot Example Keystroke with Noise/Ambience, and with Removed Ambience using Izotope RX """
###def plot_spectrogram(fname, savefigto=None):
###    width = 232.74377
###    y, sr = librosa.load(fname, sr=None, duration=0.2 + 0.01)
###    y = librosa.resample(y, sr, 44100)
###    sr = 44100
###    plt.figure(figsize=set_size(width, fraction=1))
###    plt.specgram(y, NFFT=256, Fs=sr, noverlap=128, cmap='inferno')
###    plt.xlabel('Time (s)')
###    plt.ylabel('Frequency (Hz)')
###    plt.colorbar(format='%+2.f')

###    if savefigto:
###        plt.savefig(savefigto, bbox_inches='tight', dpi=1200)

###f1 = os.path.join('dissertation_article', 'res', 'audio', 'example_keystroke_background_noise.wav')
###f2 = os.path.join('dissertation_article', 'res', 'audio', 'example_keystroke_removed_background_noise.wav')
###plot_spectrogram(f1, savefigto='dissertation_article/res/img/example_keystroke_background_noise')
###plot_spectrogram(f2, savefigto='dissertation_article/res/img/example_keystroke_removed_background_noise')
####plt.show()


##""" Plotting waveforms of same keystroke vs gcc and cc. """

##def gccphat(x, y, interp=1, max_tau=None, fs=1):
##    """https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py"""
##    n = x.shape[0] + y.shape[0]
##    x = np.fft.rfft(x, n=n)
##    y = np.fft.rfft(y, n=n)
##    R = x * np.conj(y)

##    cc = np.fft.irfft(R / np.abs(R), n=(interp*n))
##    max_shift = int(interp * n / 2)
##    if max_tau:
##        max_shift=np.minimum(int(interp * fs * max_tau), max_shift)

##    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
##    # find max cross correlation index
##    shift = np.argmax(np.abs(cc)) - max_shift
##    tau = shift / float(interp * fs)
##    return shift, cc

##def plots(f1, f2):
##    width = 232.74377
##    MAX_SHIFT = 50
##    y1, sr = librosa.load(f1, sr=None, duration=0.05)
##    y2, sr = librosa.load(f2, sr=None, duration=0.05)

##    t = np.arange(y1.shape[0])/ sr * 1000 # ms
##    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios' : [1, 2]}, figsize=set_size(width, fraction=1.5))
##    im = ax1.plot(t, y1, alpha=.5, linewidth=0.5)
##    im = ax1.plot(t, y2, alpha=.5, linewidth=0.5)
##    # ax1.fill_between(t, [-n for n in y1], y1)
##    # ax1.fill_between(t, [-n for n in y2], y2)
##    ax1.set_xlabel('Time (ms)')
##    ax1.set_ylabel('Amplitude')
##    ax1.set_xlim([t[0], t[-1]])
##    ax1.set_ylim([-1, 1])
##    ax1.xaxis.set_ticks(np.arange(min(t), max(t)+10, 10))
##    ax1.yaxis.set_ticks(np.arange(-1, 1+0.01))

##    # im = ax[1, 0].imshow(y_mfcc, aspect='auto', origin='lower', extent=[t[0], t[-1], 0, n_mfcc])
##    ax2.xaxis.set_ticks(np.arange(-70, 71, 10))
##    ax2.set_xlabel('Lag (samples)')
##    ax2.set_ylabel('CC Value')
##    # ax2.yaxis.set_visible(False)
##    ax2.yaxis.set_ticks(np.arange(0, 1+0.01, 0.25))

##    # # plt.figure(figsize=set_size(width, fraction=1))
##    # plt.figure(figsize=(set_size(width, fraction=2)[1], set_size(width, fraction=1)[0]))
##    # f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
##    # # plt.subplot(2,1,1)
##    # a0.plot(librosa.display.waveplot(y1, sr=sr, x_axis='ms', alpha=0.5))
##    # a0.plot(librosa.display.waveplot(y2, sr=sr, x_axis='ms', alpha=0.5))
##    # a0.ylabel('Amplitude')
##    # a0.ylim((-1, 1))

##    # plt.subplot(2,1,2)
##    # a1.xlabel('Lag (Samples)')
##    # a1.ylabel('Correlation')
##    y1 = librosa.util.normalize(y1) # delete?
##    y2 = librosa.util.normalize(y2)

##    # max peak
##    ax2.vlines(np.argmax(y1[288:768]) - np.argmax(y2[288:768]), 0, 1, linestyle='--', linewidth=.5, label='MPP', color='r')


##    #xcorr (normal cc)
##    ccx, lag = xcorr(y1, y2)
##    ccx = ((ccx - ccx.min()) / (ccx.max() - ccx.min()))
##    s = np.arange(-MAX_SHIFT, len(ccx) - MAX_SHIFT)
##    ax2.plot(s, ccx, alpha=1, label='CC', color='#ff7f0e', linewidth=.5)
##    ax2.plot(np.argmax(ccx) - MAX_SHIFT, np.max(ccx), marker='o', markersize=5, mfc='none', linewidth=.5)

##    # gccphat
##    shift, cc = gccphat(y1, y2, max_tau=MAX_SHIFT)
##    s = np.arange(int(-MAX_SHIFT), len(cc) - MAX_SHIFT)
##    cc = ((cc - cc.min()) / (cc.max() - cc.min()))
##    ax2.plot(s, cc, alpha=1, label='GCC-PHAT', color='#1f77b4', linewidth=.5)
##    ax2.plot(np.argmax(cc) - MAX_SHIFT, np.max(cc), marker='o', markersize=5, color='r', mfc='none', linewidth=.5)
##    fig.tight_layout()
##    leg = ax2.legend(frameon=False, loc='upper left', prop={'family':'Libertine'})
    
##def xcorr(x, y):
##    # split to lag
##    MAX_SHIFT=50
##    cc = np.correlate(x, y, "full")
##    zero_lag = np.size(cc) // 2
##    cc = cc[zero_lag-MAX_SHIFT:zero_lag+MAX_SHIFT+1]
##    # cc = np.concatenate((cc[zero_lag-MAX_SHIFT:], cc[:zero_lag+MAX_SHIFT+1]))
##    lag = np.argmax(cc) - zero_lag
##    return cc, lag
##file_directory = os.path.join('dissertation', 'res', 'audio')
##f1 = os.path.join(file_directory, 'z_nt1a_clean.wav')
##f2 = os.path.join(file_directory, 'z_sm7b_clean.wav')
##f3 = os.path.join(file_directory, 'z_nt1a_noise.wav')
##f4 = os.path.join(file_directory, 'z_sm7b_noise.wav')

### f1 = os.path.join('audio', 'test', 'p5_sm7b.wav')
### f2 = os.path.join('audio', 'test', 'p5_nt1a.wav')
##plots(f1, f2)
##savefigto = os.path.join('dissertation_article', 'res', 'img', 'clean_cc')
##plt.savefig(savefigto, bbox_inches='tight', dpi=1200)
##plots(f3, f4)
##savefigto = os.path.join('dissertation_article', 'res', 'img', 'noisy_cc')
##plt.savefig(savefigto, bbox_inches='tight', dpi=1200)
###plt.show()

#def _get_filepaths(file_directory, key):
#    filepaths = glob.iglob(os.path.join(file_directory, '**', f'{key}*.wav'), recursive=True)
#    return [filepath for filepath in filepaths]

#def plot_freq(filedir, key, freq_ratio=1, alpha=1):
#    fmin=400
#    filepaths = _get_filepaths(filedir, key)
#    magnitude_spectrum = []
#    for filepath in filepaths:
#        y, sr = librosa.load(filepath, sr=None, duration = 0.3)
#        y = librosa.util.normalize(y)
#        ft = np.fft.fft(y, sr)
#        magnitude_spectrum.append(np.abs(ft))

#    magnitude_spectrum = np.mean(magnitude_spectrum, axis=0)
#    frequency = np.linspace(fmin, sr, len(magnitude_spectrum))
#    num_frequency_bins = int(len(frequency) * freq_ratio)

#    plt.plot(frequency[:num_frequency_bins], magnitude_spectrum[:num_frequency_bins], linewidth=0.5, alpha=alpha, label=f"Average '{key}'")
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Magnitude')

#file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface', 'CH03')
#width = 455.24411
#width = 232.74377
#savefigto = os.path.join('dissertation_article', 'res', 'img', 'compare_mean_key_frequency_spectrum')
#plt.figure(figsize=set_size(width))
#plot_freq(file_directory, 'p', 14000/96000, alpha=0.8)
#plot_freq(file_directory, 'l', 14000/96000, alpha=0.8)
#plt.legend()
#plt.savefig(savefigto, bbox_inches='tight', dpi=1200)
##plt.show()

#def plot_freq2(filename, freq_ratio=1, alpha=1, label=None):
#    y, sr = librosa.load(filename, sr=None)
#    # y = librosa.util.normalize(y)
#    ft = np.fft.fft(y, sr)
#    magnitude_spectrum = np.abs(ft)
#    frequency = np.linspace(0, sr, len(magnitude_spectrum))
#    num_frequency_bins = int(len(frequency) * freq_ratio)

#    plt.plot(frequency[:num_frequency_bins], magnitude_spectrum[:num_frequency_bins], linewidth=0.5, alpha=alpha, label=label)
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Magnitude')

#f1 = os.path.join('dissertation_article', 'res', 'audio', 'Key.space35_touch.wav')
#f2 = os.path.join('dissertation_article', 'res', 'audio', 'Key.space35_delta.wav')
#f3 = os.path.join('dissertation_article', 'res', 'audio', 'Key.space35_release.wav')
#width = 232.74377
#savefigto = os.path.join('dissertation_article', 'res', 'img', 'touch_delta_release_frequency_spectrum')
#plt.figure(figsize=set_size(width))
#plot_freq2(f1, 20000/96000, label='Touch/Hit', alpha=0.8)
#plot_freq2(f2, 20000/96000, label='Delta', alpha=0.8)
#plot_freq2(f3, 20000/96000, label='Release')
#plt.legend()
#plt.savefig(savefigto, bbox_inches='tight', dpi=1200)
##plt.show()


###accuracy = [0.509, 0.314, 0.315, 0.263, 0.562, 0.340, 0.339, 0.102,
###            0.677, 0.728, 0.741, 0.703, 0.647, 0.694, 0.711, 0.670]
###precision = [0.520, 0.311, 0.311, 0.229, 0.583, 0.364, 0.364, 0.106,
###             0.694, 0.756, 0.766, 0.737, 0.657, 0.717, 0.729, 0.692]
###source = ['Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Shure SM7B',
###          'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20',
###          'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Shure SM7B',
###          'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20']
###feature = ['FFT', 'FFT', 'FFT', 'FFT', 'FFT', 'FFT', 'FFT', 'FFT',
###           'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC']
###classifier = ['RF', 'NB', 'ENS', 'SVC',
###              'RF', 'NB', 'ENS', 'SVC',
###              'RF', 'NB', 'ENS', 'SVC',
###              'RF', 'NB', 'ENS', 'SVC']

###data = list(zip(classifier, accuracy, precision, source, feature))
###df = pd.DataFrame(data, columns=['Classifier', 'Accuracy (%)', 'Precision (%)', 'Source', 'Feature']).sort_values(by=['Classifier'])
###sns.set_style("whitegrid")

###def plot_classifier_metrics(df, x, y, hue, col, aspect, hue_order=None, savefig=None):
###    width = 455.24411
###    height = set_size(width, fraction=1)[1]

###    g = sns.catplot(x=x, y=y, hue=hue, col=col, hue_order=hue_order,
###            data=df, kind='bar', height=height, aspect=aspect, saturation=.5)
###    (g.set_axis_labels('', f'{y}')
###      .set(ylim=(0, 1))
###      .set_titles('{col_name}')
###      .despine(left=True))

###    if savefig is not None:
###        g.savefig(savefig, dpi=1200, bbox_inches='tight')

#directory = os.path.join('dissertation_article', 'res', 'img')
###plot_classifier_metrics(df, x='Classifier', y='Accuracy (%)', hue='Feature', col='Source', aspect=.75,
###        savefig=os.path.join(directory, 'classifier_accuracy'))

###plot_classifier_metrics(df, x='Classifier', y='Precision (%)', hue='Feature', col='Source', aspect=.75,
###        savefig=os.path.join(directory, 'classifier_precision'))

###accuracy = [0.366, 0.301, 0.373, 0.071, 0.371, 0.302, 0.372, 0.071,
###            0.677, 0.728, 0.741, 0.703, 0.647, 0.694, 0.711, 0.670,
###            0.823, 0.729, 0.750, 0.703, 0.814, 0.700, 0.724, 0.670]
###precision = [0.320, 0.164, 0.295, 0.023, 0.320, 0.164, 0.293, 0.023,
###             0.694, 0.756, 0.766, 0.737, 0.657, 0.717, 0.729, 0.692,
###             0.844, 0.757, 0.773, 0.737, 0.833, 0.721, 0.744, 0.670]
###source = ['Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20',
###          'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20',
###          'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Shure SM7B', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20', 'Sonarworks XREF20']
###feature = ['TDoA', 'TDoA', 'TDoA', 'TDoA', 'TDoA', 'TDoA', 'TDoA', 'TDoA',
###           'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC', 'MFCC',
###           'Combined', 'Combined', 'Combined', 'Combined','Combined', 'Combined', 'Combined', 'Combined']
###classifier = ['RF', 'NB', 'ENS', 'SVC', 'RF', 'NB', 'ENS', 'SVC',
###              'RF', 'NB', 'ENS', 'SVC', 'RF', 'NB', 'ENS', 'SVC',
###              'RF', 'NB', 'ENS', 'SVC', 'RF', 'NB', 'ENS', 'SVC']

###data = list(zip(classifier, accuracy, precision, source, feature))
###df = pd.DataFrame(data, columns=['Classifier', 'Accuracy (%)', 'Precision (%)', 'Source', 'Feature']).sort_values(by=['Classifier'])

###hue_order=['TDoA', 'MFCC', 'Combined']
###plot_classifier_metrics(df, x='Classifier', y='Accuracy (%)', hue='Feature', col='Source', aspect=.75, hue_order=hue_order,
###        savefig=os.path.join(directory, 'tdoa_mfcc_combined_accuracy'))

###plot_classifier_metrics(df, x='Classifier', y='Precision (%)', hue='Feature', col='Source', aspect=.75, hue_order=hue_order,
###        savefig=os.path.join(directory, 'tdoa_mfcc_combined_precision'))



#directory = os.path.join('dissertation_article', 'res', 'img')
#plt.close('all')
#cls = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#rf1 = [0.1, 0.0, 0.04, 0.165, 0.213, 0.213, 0.15, 0.03, 0.0333, 0.0558,
#        0.035, 0.11, 0.02, 0.09, 0.385, 0.21, 0.08, 0.03, 0.02, 0.02,
#        0.135, 0.11, 0.06, 0.255, 0.598, 0.423, 0.23, 0.06, 0.0533, 0.0758]
#source = ['SM7B', 'SM7B', 'SM7B', 'SM7B', 'SM7B', 'SM7B', 'SM7B', 'SM7B', 'SM7B',
#        'XREF20', 'XREF20', 'XREF20', 'XREF20','XREF20', 'XREF20', 'XREF20', 'XREF20', 'XREF20',
#        'Cross Prediction', 'Cross Prediction', 'Cross Prediction', 'Cross Prediction', 'Cross Prediction',
#        'Cross Prediction', 'Cross Prediction', 'Cross Prediction', 'Cross Prediction']

#data = list(zip(cls, rf1, source))
#df = pd.DataFrame(data, columns=['Class', 'Probability', 'Dataset'])

#width = 232.74377
#height = set_size(width, fraction=1)[1]
## g = sns.catplot(x='Class', y='Probability', hue='Dataset', col=None, hue_order=None,
##             data=df, kind='bar', height=height, aspect=1, saturation=.5, legend_out=False, sharex=False, ci=None)

#sns.set_style("whitegrid")

#def plot_metrics(df, x, y, hue, col, col_wrap=None, ylim=(0,1), aspect=1):
#    width = 232.74377
#    height = set_size(width)[1]
#    g = sns.catplot(x=x, y=y, data=df, col=col, col_wrap=col_wrap,
#            height=height,  aspect=1.61, legend_out=False, kind='bar',
#            saturation=.5, sharex=False, ci=None, hue=hue)
#    g._legend.remove()
#    # handles, labels = g.fig.get_axes()[1].g.get_legend_handles_labels()
#    handles,labels = g.axes.flat[0].get_legend_handles_labels()
#    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3,
#            frameon=False)
#    g.set_axis_labels('', 'Probability (\%)')
#    g.set(ylim=ylim)
#    g.despine(left=True)

#    # # Add annotations.
#    # for ax in g.axes.ravel():
#    #     for c in ax.containers:
#    #         labels = [f'{(v.get_height()):.2f}' for v in c]
#    #         ax.bar_label(c, labels=labels, label_type='edge')
#    #     ax.margins(y=0.2)

#g = plot_metrics(df, x='Class', y='Probability', col=None, hue='Dataset', ylim=(0,.75))
#savename = os.path.join(directory, 'cross_prediction')
#plt.savefig(savename, dpi=1200, bbox_inches='tight')
#plt.show()
