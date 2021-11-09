import numpy as np
import librosa

class TDOA:
    
    def __init__(self, filepath, ref_filepath, max_shift, sr=None, cc_algo='cc'):
        self.filepath = filepath
        self.ref_filepath = ref_filepath
        self.max_shift = max_shift
        self.sr = sr
        self.cc_algo = cc_algo

    def get_tdoa(self):
        sig, self.sr = librosa.load(self.filepath, sr=self.sr, duration=0.15)
        ref_sig, _ = librosa.load(self.ref_filepath, sr=self.sr, duration=0.15)

        if self.cc_algo == 'cc':
            tau, _ = self._cc(sig, ref_sig)
        elif self.cc_algo == 'gccphat':
            tau, _ = self._gccphat(sig, ref_sig)
        return tau / self.sr # units in seconds.

    def _cc(self, sig, ref_sig):
        """
        Correlation algorithm as generally defined in signal processing texts
        [1]. This method assumes that sig and ref_sig come from the same
        source.

        Parameters
        ----------
        sig : np.array
            Audio time series. The first signal in the cross correlation
            algorithm.

        refsig: np.array
            Audio time series. The reference signal in the cross correlation
            algorithm.

        Returns
        -------
        cc : np.array
            An array of cross correlation values between -max_shift and
            max_shift. Useful for plotting the correlation.

        tau : float
            The time delay between sig and ref_sig in samples.

        [1] https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
        """

        cc = np.correlate(sig, ref_sig, 'full')
        zero_lag = np.size(cc) // 2
        cc = cc[zero_lag-self.max_shift : zero_lag+self.max_shift+1]
        cc = ((cc - cc.min()) / (cc.max() - cc.min()))
        tau = np.argmax(cc) - self.max_shift # zero_lag
        return tau, cc

    def _gccphat(self, sig, ref_sig):
        """
        Generalised cross-correlation phase transform (GCC-PHAT) algorithm.
        This method assumes that sig and ref_sig come from the same source.
        Slightly modified from [1].

        Parameters
        ----------
        sig : np.array
            Audio time series. The first signal in the cross correlation
            algorithm.

        refsig: np.array
            Audio time series. The reference signal in the cross correlation
            algorithm.

        Returns
        -------
        cc : np.array
            An array of cross correlation values between -max_shift and
            max_shift. Useful for plotting the correlation.

        tau : float
            The time delay between sig and ref_sig in samples.

        [1] xiongyihui GitHub
            https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
        """

        n = sig.shape[0] + ref_sig.shape[0]
        sig = np.fft.rfft(sig, n=n)
        ref_sig = np.fft.rfft(ref_sig, n=n)
        R = sig * np.conj(ref_sig)

        cc = np.fft.irfft(R / np.abs(R), n=n)
        max_shift = int(n / 2)
        if self.max_shift:
            max_shift = np.minimum(int(self.sr*self.max_shift), self.max_shift)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
        # find max cross correlation index
        shift_plot = np.arange(-max_shift, max_shift+1)
        tau = np.argmax(cc) - max_shift
        return tau, cc

