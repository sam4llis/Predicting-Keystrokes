import matplotlib as mpl
new_rc_params = {'text.usetex': True,
         'svg.fonttype': 'none',
         'text.latex.preamble': r'\usepackage{libertine}',
         'font.size': 7,
         'font.family': 'Linux Libertine',
         'mathtext.fontset': 'custom',
         'mathtext.rm': 'libertine',
         'mathtext.it': 'libertine:italic',
         'mathtext.bf': 'libertine:bold',
         'axes.linewidth': 0.1,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7,
         'hatch.linewidth': 0.01,
         'legend.fontsize':7,
         'legend.handlelength': 2
         }

mpl.rcParams.update(new_rc_params)
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')
import librosa
import librosa.display
import os
# from figures import set_size

def gccphat(x, y, interp=1, max_tau=50/96000, fs=96000):
    """https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py"""
    # to avoid time domain aliasing, use FFT size 'n' as large as shapes of x and y
    n = x.shape[0] + y.shape[0]
    x = np.fft.rfft(x, n=n)
    y = np.fft.rfft(y, n=n)
    R = x * np.conj(y)

    cc = np.fft.irfft(R / np.abs(R), n=(interp*n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift=np.minimum(int(interp*fs*max_tau), max_shift)
    # print(max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    # find max cross correlation indexo
    shift_plot = np.arange(-max_shift, max_shift+1)
    shift = np.argmax(cc) - max_shift
    # print(shift)
    tau = shift / fs
    return tau, cc

def get_plot_gccphat(filename1, filename2):
    y1, sr1 = librosa.load(filename1, sr=None, duration=0.03)
    y2, sr2 = librosa.load(filename2, sr=None, duration=0.03)
    y1 = librosa.util.normalize(y1)
    y2 = librosa.util.normalize(y2)

    tau, cc = gccphat(y1, y2)

    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    label = os.path.splitext(os.path.split(filename1)[1])[0]
    librosa.display.waveplot(y1, sr=sr1, alpha=.5, label=label)
    label = os.path.splitext(os.path.split(filename2)[1])[0]
    librosa.display.waveplot(y2, sr=sr2, alpha=.5, label=label)
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Lag (Samples)')
    plt.ylabel('Correlation')
    max_shift=75
    # max_shift = int((y1.shape[0] + y2.shape[0]) / 2)
    t = np.arange(-max_shift, max_shift+1)
    # t = np.arange(0, cc.shape[0])
    plt.plot(t, cc)
    tau = tau*sr1
    # print(f'tau = {tau} samples')
    plt.plot(np.around(tau), np.max(cc), marker='o', markersize=5, color='r', mfc='none')



def tdoa(tau, sr, delta_d_mics, key):
    tau = tau / sr
    V_LIGHT = 343 # m/s
    # tau = tau / sr # s
    AB_star = tau * V_LIGHT # m
    x_b = delta_d_mics / 2 # m

    x_more_than = (-1 * (AB_star**2 * (AB_star**2 - 4 * x_b**2)) / (4 * (4 * x_b**2 - AB_star**2)) )**0.5
    x_plot = np.arange(x_more_than+0.001, 0.10, 0.001)
    y_plot = np.array([])

    for x in x_plot:
        y = ( AB_star**2 / 4 - x_b**2 + x**2 * ((4 * x_b**2) / AB_star**2 - 1) )**0.5
        y_plot = np.append(y_plot, y)

    x_vals = np.where(x_plot > 0.02)[0]
    ar = np.where(x_plot>0.02)
    y_vals = y_plot[ar]
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    # slope = np.mean(np.gradient(x_vals, y_vals))
    slope = np.nanmean(dy/dx) * 1000
    # if AB_star >= 0:
    #     pass
    # else:
    #     slope = -slope
    alpha_star = np.arctan(slope)
    alpha = np.degrees(np.pi/2 - alpha_star)

    # if slope > 0:
    #     alpha = np.degrees(np.pi/2 - np.arctan(slope))
    # elif slope < 0:
    #     alpha = np.degrees(np.pi/2 - np.arctan(slope))

    if tau > 0:
        alpha = -alpha
    if tau == 0:
        alpha = 0
    # if slope == np.nan:
    #     alpha=0
    # if alpha == np.nan:
    #     alpha=0
    # if np.absolute(tau*96000) >= 50:
    #     alpha=0
    # # print(key)
    # # print(y_plot)
    # # print(tau)
    # # print(slope)
    # # print(alpha)
    # # input()

    # if AB_star >= 0:
    #     alpha = np.degrees(np.pi/2 - alpha_star)
    # else:
    #     alpha = np.degrees(-(np.pi/2) - alpha_star)

    plt.plot(x_plot, y_plot, label=f'{key}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.text(0.101, y_plot[-1], f'{key}')
    # print(f'tau: {tau}')
    # print(f'AB*: {np.round(AB_star * 100, 3)} cm')
    # print(f'Gradient: {np.round(slope, 3)}')
    # print(f'alpha*: {np.round(np.degrees(alpha_star), 3)}')
    # print(f'alpha: {np.round(alpha, 3)}')
    return alpha

def get_tau_from_files(f1, f2, key):
    y1, sr1 = librosa.load(f1, sr=None)
    y2, sr2 = librosa.load(f2, sr=None)
    tau, _ = gccphat(y1, y2)
    return tau
    # get_plot_gccphat(f1, f2)
    # print(f'{key} KEY')
    # return tdoa(tau, sr1, 0.300, key=f'{key}')

def plot_keys(keys):
    plt.figure(figsize=(20,20))
    # alpha, k = [], []
    tau, k = [],[]
    for key in keys:
        for i in range(14):
            f1 = os.path.join('audio', 'Interface', 'Shure SM7B1', key, f'{key}{i}.wav')
            # f1 = os.path.join('audio', 'Interface', 'Shure SM7B', f'{key}{i}.wav')
            f2 = os.path.join('audio', 'Interface', 'Rode NT1A1', key, f'{key}{i}.wav')
            # f2 = os.path.join('audio', 'Interface', 'Rode NT1A', f'{key}{i}.wav')
            # alpha.append(get_tau_from_files(f1, f2, key=key))
            tau.append(get_tau_from_files(f1, f2, key=key))
            k.append(key)
    # plt.figure(figsize=(5,5))
    # plt.bar([key for key in keys], alpha)
    # return alpha, k
    return tau, k

def xcorr(x, y):
    # split to lag
    MAX_SHIFT=50
    cc = np.correlate(x, y, "full")

    zero_lag = np.size(cc) // 2
    cc = cc[zero_lag-MAX_SHIFT:zero_lag+MAX_SHIFT+1]
    # cc = np.concatenate((cc[zero_lag-MAX_SHIFT:], cc[:zero_lag+MAX_SHIFT+1]))
    lag = np.argmax(cc) - zero_lag
    return cc, lag

def plots(f1, f2):
    MAX_SHIFT = 50
    y1, sr = librosa.load(f1, sr=None)
    y2, sr = librosa.load(f2, sr=None)

    ccx, lag = xcorr(y1, y2)
    ccx = ((ccx - ccx.min()) / (ccx.max() - ccx.min()))
    s = np.arange(-MAX_SHIFT, len(ccx) - MAX_SHIFT)
    tau = np.argmax(ccx) - MAX_SHIFT
    return tau

def plot_keys(keys):
    tau1, tau2, tau3, k, alpha = [], [], [], [], []
    for key in keys:
        for i in range(30):
            try:
                f1 = os.path.join('code', 'src', 'data', 'audio', 'Interface_new_96khz', 'Shure SM7B', f'{key}', f'{key}{i}.wav')
                f2 = os.path.join('code', 'src', 'data', 'audio', 'Interface_new_96khz', 'Rode NT1A', f'{key}', f'{key}{i}.wav')
            except FileNotFoundError:
                continue
            tau1.append(plots(f1, f2))
            y1, _ = librosa.load(f1, sr=None, duration=0.15)
            y2, _ = librosa.load(f2, sr=None, duration=0.15)
            tau2.append(gccphat(y1, y2)[0]*96000)
            tau3.append(np.argmax(y2) - np.argmax(y1))
            # print(f'{key}{i}.wav')
            # print(plots(f1, f2))
            k.append(key)
            alpha.append(tdoa(tau=plots(f1, f2), sr=96000, delta_d_mics= 0.30, key=key))
    return tau1, tau2, tau3, k, alpha

tau1, tau2, tau3, keys, alpha = plot_keys('1qw')
plt.xlim(0, 0.050)
print(tau3)
print(f'Keys Analysed: {len(tau1)}')
print()
print(f'(CC) Mean Sample Lag: {np.round(np.mean(tau1), 1)}')
print(f'STD: {np.round(np.std(tau1), 3)}')
print()
print(f'(GCC-PHAT) Mean Sample Lag: {np.round(np.mean(tau2), 1)}')
print(f'STD: {np.round(np.std(tau2), 3)}')
print()
print(f'(MPP) Mean Sample Lag: {np.round(np.mean(tau3), 1)}')
print(f'STD: {np.round(np.std(tau3), 3)}')
plt.show()



# a,_,_, keys, alpha = plot_keys('asdfghjkl')
# # get_plot_gccphat('audio/Interface/Shure SM7B/a5.wav', 'audio/Interface/Rode NT1A/a5.wav')
# # plt.show()
# b = [0] * len(a)
# plt.figure(figsize=(10,10))
# for i in range(len(a)):
#     plt.scatter(a[i], alpha[i], color='b')
#     plt.annotate(keys[i], (a[i], 0))
# plt.show()

# from sklearn.cluster import KMeans
# # x = [1,5,1.5,8,1,9]
# # y = [2,8,1.8,8,0.6,11]

# # X = np.array(list(zip(a, [0] * len(a))))
# X = np.array(list(zip(a, alpha)))
# print(X)
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)

# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
# print(centroids)
# print(labels)
# colors = ['r.','g.','b.','c.','k.','y.', 'm.'] * 10

# for i in range(len(X)):
#     # print(f'Coordinate: {X[i]}', f'Label: {labels[i]}')
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
#     plt.annotate(keys[i], (a[i], np.random.uniform(low=X[i][1], high=X[i][1]+0.5, size=(1,))[0]))

# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10)
# x = kmeans.predict([[20,0], [-20,0]])
# print(x[0])
# plt.show()

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

width = 455.24411
width = 232.74377
plt.figure(figsize=set_size(width))
parent_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface')

f1 = os.path.join(parent_directory, 'Shure SM7B','q', 'q2.wav')
f2 = os.path.join(parent_directory, 'Rode NT1A', 'sam','q2.wav')
plt.plot(tdoa(plots(f1, f2), 96000, 0.279, 'q'))

# f1 = os.path.join(parent_directory, 'Shure SM7B', '11.wav')
# f2 = os.path.join(parent_directory, 'Rode NT1A', '11.wav')
# plt.plot(tdoa(plots(f1, f2), 96000, 0.279, '1'))

print()

f1 = os.path.join(parent_directory, 'Shure SM7B', 't', 't15.wav')
f2 = os.path.join(parent_directory, 'Rode NT1A', 'sam', 't15.wav')
plt.plot(tdoa(plots(f1, f2), 96000, 0.279, 't'))


# f1 = os.path.join(parent_directory, 'Shure SM7B', '51.wav')
# f2 = os.path.join(parent_directory, 'Rode NT1A', '51.wav')
# plt.plot(tdoa(plots(f1, f2), 96000, 0.279, '5'))

print()

f1 = os.path.join(parent_directory, 'Shure SM7B', 'p', 'p1.wav')
f2 = os.path.join(parent_directory, 'Rode NT1A', 'sam','p1.wav')
plt.plot(tdoa(plots(f1, f2), 96000, 0.279, 'p'))

# f1 = os.path.join(parent_directory, 'Shure SM7B', '01.wav')
# f2 = os.path.join(parent_directory, 'Rode NT1A', '01.wav')
# plt.plot(tdoa(plots(f1, f2), 96000, 0.279, '0'))


# plt.legend()
plt.ylim(0, 4.5)

# plt.ylim()


fpath = os.path.join('dissertation_article', 'res', 'img', 'tdoa_x_and_y')
plt.savefig(fpath, bbox_inches='tight', dpi=1200)
plt.show()

def avg_angle(key):

    angles = []
    for i in range(50):
        f1 = os.path.join(parent_directory, 'Shure SM7B', f'{key}{i}.wav')
        f2 = os.path.join(parent_directory, 'Rode NT1A', f'{key}{i}.wav')
        angles.append(tdoa(plots(f1, f2), 96000, 0.279, key))
    return np.round(np.mean(angles), 1)







