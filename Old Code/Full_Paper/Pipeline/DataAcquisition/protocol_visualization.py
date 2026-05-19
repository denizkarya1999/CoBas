import numpy as np
import matplotlib.pyplot as plt
import wave

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr

x, sr = load_wav("5_15sPause_BeaconProtocol.wav")

t = np.arange(len(x)) / sr

plt.figure(figsize=(14, 4))
plt.plot(t, x, linewidth=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Full waveform overview")
plt.tight_layout()
plt.show()




# import numpy as np
# import wave
#
# def load_wav(path):
#     with wave.open(path, 'rb') as wf:
#         sr = wf.getframerate()
#         x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
#     return x.astype(np.float32), sr
#
# x, sr = load_wav("your_file.wav")
#
# # extract BEGIN and END regions manually (adjust times)
# begin = x[int(60*sr):int(62*sr)]
# end   = x[int((len(x)/sr - 7)*sr):int((len(x)/sr - 5)*sr)]
#
# print("Max abs difference:", np.max(np.abs(begin - end)))
# print("RMS BEGIN:", np.sqrt(np.mean(begin**2)))
# print("RMS END:  ", np.sqrt(np.mean(end**2)))
#
