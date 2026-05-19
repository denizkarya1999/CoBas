import numpy as np
import glob
import subprocess
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate

# BEACON PROTOCOL
# AUDIO EXTRACTOR
# PHASE A ->
# PHASE B ->
# PHASE C -> 

def _audioExtractor(video_path: str, audio_output_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            audio_output_path
        ],
        check=True
    )
    
def extractAudio(path: str, path_is_dir: bool = True):
    
    if path_is_dir:
        video_files = glob.glob(path + "/*.MOV")
        # print(video_files)
        
        for path in video_files:
            output_path = path[:-3] + "wav"
            print(output_path)
            _audioExtractor(video_path=path, audio_output_path=output_path)
            
def visualize(x, y, save: bool = True):
    plt.plot(x[:20000], label='50soc')
    plt.plot(y[:20000], label='100soc')
    plt.legend()
    if save:
        image_name = "50soc_100soc.png"
        plt.savefig("/Users/pedropaiva/Dev/Research/CoBasE-Energy/cobasFork/Full_Paper/AcousticTests/NewMetrics/croppingAndAlignmentTest_phone_000/cropped_videos/PostBeaconComparison/" + image_name)
    plt.show()
    
def phaseA(x, y, fs):
    # --- Step 1: Convert to float ---
    x = x.astype(float)
    y = y.astype(float)

    # --- Step 2: Remove DC offset ---
    x = x - np.mean(x)
    y = y - np.mean(y)

    # --- Step 3: Normalize amplitude ---
    x = x / np.std(x)
    y = y / np.std(y)

    # --- Step 4: Cross-correlation ---
    corr = correlate(y, x, mode='full')

    # --- Step 5: Lag indices ---
    lags = np.arange(-len(x) + 1, len(x))

    # --- Step 6: Edge handling (center window) ---
    center = len(corr) // 2
    window = int(0.1 * len(corr))  # 10% around center

    sub_corr = corr[center - window : center + window]
    sub_lags = lags[center - window : center + window]

    # --- Step 7: Find best lag ---
    lag = sub_lags[np.argmax(sub_corr)]

    # --- Step 8: Align signals ---
    if lag > 0:
        y_aligned = y[lag:]
        x_aligned = x[:len(y_aligned)]
    else:
        x_aligned = x[-lag:]
        y_aligned = y[:len(x_aligned)]

    return x_aligned, y_aligned, lag, lag / fs
                        
if __name__ == "__main__":
    
    INPUT_DIR = "/Users/pedropaiva/Dev/Research/CoBasE-Energy/cobasFork/Full_Paper/AcousticTests/NewMetrics/croppingAndAlignmentTest_phone_000/cropped_videos"
    
    # extractAudio(path=INPUT_DIR)
        
    fs, x = wavfile.read(INPUT_DIR + "/cropped_control.wav")
    fs, y = wavfile.read(INPUT_DIR + "/cropped_0soc.wav")
    
    print(fs)
    print(f'lenx: {len(x)}')    
    print(f'leny: {len(y)}')    
    
    x_aligned, y_aligned, lag, n = phaseA(x, y, fs)
    
    print(fs)
    print(f'lenx_aligned: {len(x_aligned)}')    
    print(f'leny_aligned: {len(y_aligned)}')    
    
    # plt.plot(x)
    # image_name = "100soc.png"
    # plt.savefig("/Users/pedropaiva/Dev/Research/CoBasE-Energy/cobasFork/Full_Paper/AcousticTests/NewMetrics/croppingAndAlignmentTest_phone_000/cropped_videos/PreBeacon/" + image_name)
    # plt.show()

    visualize(x_aligned, y_aligned)



    
    

