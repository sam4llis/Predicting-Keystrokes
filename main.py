import asyncio
import os
import sys
from recorder import *
from keylogger import *
from slice_audio import *
from stream_latency import *

async def main():
    fname1 = os.path.join('code', 'src', 'data', 'audio', 'MacBook.wav')
    fname2 = os.path.join('code', 'src', 'data', 'audio', 'Interface.wav')
    # instantiate recording/logging
    r1 = RecordSession('MacBook Pro Microphone', fname1, rate=96000)
    r2 = RecordSession('Universal Audio Thunderbolt', fname2, mono=False, rate=192000)
    l = KeyloggerSession()

    with r1.start_session() as r1, r2.start_session() as r2:
    # with r2.start_session() as r2:
        # start and stop recording/logging concurrently
        await asyncio.gather(r1.start_recording(), r2.start_recording(), l.start_logging())
        # await asyncio.gather(r2.start_recording(), l.start_logging())
        await asyncio.gather(r1.stop_recording(), r2.stop_recording())
        # await asyncio.gather(r2.stop_recording())

        input('\nPress Enter After Post-Processing Audio')
        l.json_to_file(out_f=os.path.join('code', 'src', 'data', 'Main_Boxing3.json'))
        # stream_latency = get_mean_latency(fname1, fname2)
        # print(f'stream latency = {stream_latency*1000} ms')

        # slice audio file into its respective keystrokes
        # s1 = Slice(fname1, l.to_dict())

        # s2 = Slice(fname2, l.to_dict())
        # s1.slice()
        # s2.slice(stream_latency=stream_latency)
        # s2.slice(stream_latency=-0.023)

async def main1():
    fname1 = os.path.join('code', 'src', 'data', 'audio', 'SAMPLE_TEXT-USER02.wav')
    # instantiate recording/logging
    r = RecordSession('MacBook Apollo Aggregate', fname1, rate=96000, mono=False)
    l = KeyloggerSession()

    with r.start_session() as r:
        # start and stop recording/logging concurrently
        await asyncio.gather(r.start_recording(), l.start_logging())
        await asyncio.gather(r.stop_recording())
        l.json_to_file(out_f=os.path.join('code', 'src', 'data', 'SAMPLE_TEXT-USER02.json'))

if __name__ == "__main__":
    asyncio.run(main1())

