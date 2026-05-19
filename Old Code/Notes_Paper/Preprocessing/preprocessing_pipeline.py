from fast_split import fast_split
from frame_cropping import crop_all_images
from stft_fft_librosa import compute_frequency_domain_spectrogram
from tqdm import tqdm
import numpy as np
import glob, os

if __name__ == "__main__":

    #======= CONFIGURATIONS =======
    #=== PATHS ===
    RAW_VIDEOS_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/official00/test/crpd_raw_videos"

    AUDIO_SEGMENTS_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/official00/test/audio_segments"
    VIDEO_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/Test4/video_frames"

    CROPPED_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/Test0/cropped_frames"
    FFT_SPEC_ARRAYS_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/official00/test/test_ffts"

    #=== Short-Time Fourier Transform (STFT --> FFT) ===
    SAMPLING_RATE = 48000
    N_FFT = 2048
    HOP_LENGTH = 240
    WIN_LENGTH = 480
    FMIN = 15000
    FMAX = 19200

    #======= PREPROCESSING =======

    if not os.path.isdir(RAW_VIDEOS_DIR):
        print("[ERROR]: NO RAW VIDEOS DIR FOUND!")

    os.makedirs(AUDIO_SEGMENTS_DIR, exist_ok=True)
    os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)
    os.makedirs(CROPPED_FRAMES_DIR, exist_ok=True)
    os.makedirs(FFT_SPEC_ARRAYS_DIR, exist_ok=True)

    raw_video_files = glob.glob(RAW_VIDEOS_DIR + "/*")

    #=== AUDIO/FRAME CHUNK SPLIT ===
    print("[UPDATE]: AUDIO/FRAME EXTRACTION INITIALIZED!")
    for video_file in tqdm(raw_video_files):
        fast_split(video_path=video_file,
                   out_audio_dir=AUDIO_SEGMENTS_DIR,
                   out_frame_dir=VIDEO_FRAMES_DIR)

    #=== VIDEO FRAME CROPPING ===
    # print("[UPDATE]: FRAME CROPPING INITIALIZED!")
    # crop_all_images(extracted_frames_dir=VIDEO_FRAMES_DIR,
    #                 output_dir=CROPPED_FRAMES_DIR,
    #                 verbose=False,
    #                 chunk=10)

    #=== FREQUENCY-DOMAIN SPEC. ARRAY COMPUTATION ===
    audio_segment_files = glob.glob(AUDIO_SEGMENTS_DIR + "/*")
    print("[UPDATE]: STFT/FFT COMPUTE INITIALIZED!")
    for segment_file_path in tqdm(audio_segment_files):
        S_ultra = compute_frequency_domain_spectrogram(
            audio_path=segment_file_path,
            sampling_rate=SAMPLING_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=FMIN,
            fmax=FMAX)

        file_name = segment_file_path.split('/')[-1].split('.')[0] + ".npy"
        file_name = FFT_SPEC_ARRAYS_DIR + '/' + file_name
        np.save(file_name, S_ultra.astype(np.float32))
