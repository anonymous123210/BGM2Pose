import wave
import pyaudio
import os


def printWaveInfo(wf):
    print("Channel Num:", wf.getnchannels())
    print("Sample width:", wf.getsampwidth())
    print("Sampling rate:", wf.getframerate())
    print("Frame num:", wf.getnframes())
    print("Parameters:", wf.getparams())
    print("Duration (seconds):", float(wf.getnframes()) / wf.getframerate())


if __name__ == "__main__":
    wf = wave.open(os.path.join("data/Mic/210528_001.WAV"), "r")

    printWaveInfo(wf)

    # p = pyaudio.PyAudio()
    # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                 channels=wf.getnchannels(),
    #                 rate=wf.getframerate(),
    #                 output=True)

    # chunk = 1024
    # data = wf.readframes(chunk)
    # while data != '':
    #     stream.write(data)
    #     data = wf.readframes(chunk)
    # stream.close()
    # p.terminate()
