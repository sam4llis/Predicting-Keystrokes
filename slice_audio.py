import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import json
from keylogger import *
import stream_latency
import utils

# CUSTOM PLOT FORMATTING FOR REPORT - Remove if not required.
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
                 'legend.handlelength': 2}

mpl.rcParams.update(new_rc_params)

class Slice:

    def __init__(self, filename, key_dict): #TODO: incorporate latency! and get TDoA measurements here (likely before splitting on silence
        self.filename = filename
        self.key_dict = key_dict
        self.events = self.key_dict.events

    def slice(self, stream_latency=0):
        self.signal = AudioSegment.from_file(self.filename)
        self.mono = True if self.signal.channels == 1 else False

        for i, event in enumerate(self.events):
            if event.event_type == 'KEY_DOWN':
                i += 1
                while self.events[i].event_type == 'KEY_DOWN' or self.events[i].char != event.char:
                    i += 1
                latency_ms = 40
                key_start_ms = (event.delta_time / 1E+06) - stream_latency - latency_ms
                key_end_ms = (self.events[i].delta_time / 1E+06) + 100
                # key_end_ms = key_start_ms + 200
                key_slice = self.signal[key_start_ms-30:key_end_ms]
                key = event.char.replace("'","")

                if self.mono:
                    self.filepath = self._get_slice_filepath(key)
                    key_slice.export(self.filepath, format='wav', bitrate='96k')
                    self.process_slice(self.filepath)
                else:
                    # stereo compatibility
                    samples = key_slice.get_array_of_samples()
                    mono_channels = []

                    for i in range(key_slice.channels):
                        samples_current_channel = samples[i::key_slice.channels]

                        try:
                            mono_data = samples_current_channel.tobytes()
                        except AttributeError:
                            mono_data = samples_current_channel.tostring()

                        mono_channels.append(key_slice._spawn(mono_data,
                            overrides={'channels': 1, 'frame_width': self.signal.sample_width}))

                    for i, channel in enumerate(mono_channels):
                        self.filepath = self._get_slice_filepath(key, indev_name=f'Channel {i}')
                        channel.export(self.filepath, format='wav')
                        self.process_slice(self.filepath)

    def process_slice(self, audiosegment):
        audio_util = AudioUtil(audiosegment)
        audio_util.normalize()
        # audio_util.remove_silence()
        audio_util.export()

    def _get_slice_filepath(self, key, indev_name=''):
        util = FileUtil(key)
        if self.mono:
            filepath = util.iterate_file(os.path.splitext(self.filename)[0])
            return filepath
        else:
            filepath = util.iterate_file(os.path.splitext(self.filename)[0], indev_name)
            return filepath

class AudioUtil:

    def __init__(self, filepath):
        self.filepath = filepath
        self.y, self.sr = librosa.load(self.filepath, sr=None)
        self.l1 = librosa.get_duration(self.y, self.sr)

    def normalize(self):
        self.y = librosa.util.normalize(self.y)

    def export(self):
        sf.write(file=self.filepath, data=self.y, samplerate=self.sr, format='WAV', subtype='PCM_24')


    def audiosegment_to_librosa(self):
        """
        Converts a pydub AudioSegment to librosa format.
        Adapted from code by Edresson Casanova [1] and pydub official API documentation [2].

        Parameters
        ----------
        audiosegment : pydub.audio_segment.AudioSegment
            An AudioSegment object from the pydub library.

        Returns
        -------
        np.array
            A librosa-formatted numpy array containing an audio time series.

        [1] https://stackoverflow.com/questions/62916406/audiosegment-to-librosa
        [2] https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        """

        audio = self.audiosegment.split_to_mono()
        samples = [s.get_array_of_samples() for s in audio]
        arr = np.array(samples).T.astype(np.float32)#/2**31
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)
        # arr = librosa.core.resample(y=arr, orig_sr=self.sr, target_sr= self.sr, res_type='kaiser_best')
        return arr

    def refine_keystroke(self, mono=True):
        print(self.filepath)
        signal = AudioSegment.from_file(self.filepath)
        # peak = stream_latency.get_transient(self.y, self.sr) * 1000 # in ms
        peaks = stream_latency.get_onset_times(self.y, self.sr)
        if len(peaks) == 0:
            peak=15
        else:
            peak = peaks[0] * 1000 # ms
        
        # peaks = stream_latency.get_onset_times(self.y, self.sr)[0]* 1000
        refined_signal = signal[peak - 15: peak + 135]
        refined_signal.export(self.filepath, format='wav')
        y, sr = librosa.load(self.filepath, sr=None)
        sf.write(file=self.filepath, data=y, samplerate=sr, format='WAV', subtype='PCM_24')

        if not mono:
            ref_filepaths = utils._get_ref_filepath(self.filepath, to_string=False)

            for ref in ref_filepaths:
                if 'CH03' in utils.splitall(ref)[-3]:
                    ref_filepath = ref
            
            print(ref_filepath)
            ref_signal = AudioSegment.from_file(ref_filepath)
            refined_ref_signal = ref_signal[peak - 15: peak + 135]

            refined_ref_signal.export(ref_filepath, format='wav')
            y, sr = librosa.load(ref_filepath, sr=None)
            sf.write(file=ref_filepath, data=y, samplerate=sr, format='WAV', subtype='PCM_24')

    def remove_silence(self):
        S = np.abs(librosa.stft(self.y))
        rmse = librosa.feature.rms(y=self.y, frame_length=64, hop_length=16)[0]
        self.y, index = librosa.effects.trim(self.y, top_db=10, ref=rmse, frame_length=64, hop_length=16)
        self.l2 = librosa.get_duration(self.y, self.sr)
        print(f'Time Removed = {(self.l1 - self.l2) * 1000} ms')

class FileUtil:

    def __init__(self, key, file_format='wav'):
        self.key = key
        self.file_format = file_format

    def _create_directory(self, *args):
        self.dir = os.path.join(*args, self.key)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def iterate_file(self, *args):
        i = 0
        self._create_directory(*args)
        self.filename = f'{self.key}{i}.{self.file_format}'
        while os.path.exists(os.path.join(self.dir, self.filename)):
            i += 1
            self.filename = f'{self.key}{i}.{self.file_format}'
        return os.path.join(self.dir, self.filename)

class DotDict(dict):
    """Inspired from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__


def main():
    # Import JSON into a python dictionary.
    par_dir = os.path.join('code', 'src', 'data', 'audio', 'USER02', '{PASTE FOLDER HERE}')
    fname_json = os.path.join(par_dir, 'UNIX.json')
    d = KeyloggerSession().import_json(out_f=fname_json)

    fname_ch1 = os.path.join(par_dir, 'CH01.wav')
    fname_ch2 = os.path.join(par_dir, 'CH02.wav')
    fname_ch3 = os.path.join(par_dir, 'CH03.wav')
    s1 = Slice(fname_ch1, d)
    s2 = Slice(fname_ch2, d)
    s3 = Slice(fname_ch3, d)

    # Slice audio into respective keystrokes.
    s1.slice(stream_latency=-0.032) # MacBook
    s2.slice(stream_latency=0) # XREF20
    s3.slice(stream_latency=0) # Shure SM7B

    # Refine sliced audio.
    directory_ch1 = os.path.join(par_dir, 'CH01')
    filepaths = utils._get_filepaths(directory_ch1)
    for filepath in filepaths:
        audio = AudioUtil(filepath)
        audio.refine_keystroke(mono=True)

    directory_ch2 = os.path.join(par_dir, 'CH02')
    filepaths = utils._get_filepaths(directory_ch2)
    for filepath in filepaths:
        audio = AudioUtil(filepath)
        audio.refine_keystroke(mono=False)

if __name__ == "__main__":
    main()
