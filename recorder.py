import asyncio
import pyaudio
import wave

class RecordSession:
    """
    Starts a session to record audio to a specified WAV file:

    indev : str
        Input device used when recording.

    f_out : str
        The path at which the WAV audio file will be written.

    mono : bool, default: True
        True if the input is a mono signal, and False otherwise.

    rate : int, default: 96000
        The sample rate of the recorded audio.
    """

    def __init__(self, indev, f_out, mono=True, rate=96000):
        self.format = pyaudio.paInt32
        self.ch = 1 if mono else 2
        self.rate = rate
        self.chunk = 1024

        self.indev = indev
        self.f_out = f_out

    def start_session(self):
        """ Begins an audio recording. """
        return Record(self.indev, self.f_out, self.ch, self.rate, self.chunk)

class Record:
    """ Records audio in a non-blocking format. """

    def __init__(self, indev, f_out, channels, rate, chunk):
        self.indev = indev
        self.f_out = f_out
        self.format = pyaudio.paInt32
        self.ch = channels
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.wf = self.make_file(self.f_out, 'wb')
        self.stream = None

    def __enter__(self):
        if not self.is_valid_input():
            print(f"Device '{self.indev}' is not valid. Try:")
            self.show_input_devices()
            raise
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def show_input_devices(self):
        """ Prints all possible recording devices on the users system. """
        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_host_api_device_index(0, i)
            if device.get('maxInputChannels') > 0:
                print(f"{device.get('name')}")

    def is_valid_input(self):
        """ Checks if the user-specified input device is valid. """
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev.get('name') == self.indev:
                self.dev_index = i
                return True
        else:
            return False

    async def start_stream(self):
        self.stream.start_stream()

    async def start_recording(self):
        """ Begin an (asynchronous) audio recording. """
        self.stream = self.p.open(format = self.format,
                                  channels = self.ch,
                                  rate = self.rate,
                                  input = True,
                                  input_device_index = self.dev_index,
                                  frames_per_buffer = self.chunk,
                                  stream_callback = self.stream_callback())
        await self.start_stream()

    def stream_callback(self):
        """ Create audio streams in a non-blocking format. """
        def callback(in_data, frame_count, time_info, status):
            data = self.wf.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback

    async def stop_recording(self):
        """ Stop recording audio. """
        self.stop_recording_callback()

    def stop_recording_callback(self):
        self.stream.stop_stream()

    def close(self):
        self.stream.close()
        self.p.terminate()
        self.wf.close()

    def make_file(self, filename, fmode='wb'):
        """
        Writes audio data to a user-specified filepath.

        Parameters
        ----------
        filename: str
            The name of the file to which the audio data is written.

        fmode: str, default:wb
            The type of mode to open the file, defaults to 'write binary' mode.
        """
        wf = wave.open(filename, fmode)
        wf.setnchannels(self.ch)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        return wf

