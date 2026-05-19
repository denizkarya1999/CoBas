# Battery State Classification using Acoustic Reflection and Image Fusion

## Overview
This project investigates how **acoustic reflections** and **visual cues** can be used together to determine the **State of Charge (SoC)** of a lithium-ion battery.

The system emits a near-ultrasonic waveform (15–19 kHz) from a laptop speaker and records the reflected signal using a smartphone microphone while also capturing images of the battery.  
A **MultiModal ResNet** model then fuses both **audio (Mel-spectrogram)** and **image (RGB)** features to classify batteries into:
> **Full · Half · Empty**

---

## Key Features
✅ Multimodal deep-learning model (audio + image fusion)  
✅ Band-pass filtering (15–19 kHz) and spectral subtraction for noise reduction  
✅ Automatic data extraction from `.MOV` videos (audio + image frames)  
✅ Training visualization (t-SNE feature separation)  
✅ Segment-level and video-level predictions with confusion matrix  

---
## Experimental Setup

- **Signal source:** MacBook speaker (plays the generated WAV file).  
- **Targets:** Water bottles (full, half-full, empty).  
- **Placement:** Bottles placed **10–15 cm** to the right of the MacBook speaker.  
- **Recording device:** iPhone microphone placed close to the bottle; recorded at **different angles** (tilled the bottle at different angle).  
- **Procedure (per recording):**
  1. Place bottle 10–15 cm right of MacBook speaker.  
  2. Position iPhone mic close to the bottle at chosen angle.  
  3. Start recording on iPhone.  
  4. Play the WAV file from the MacBook speaker.  
  5. Stop recording after the playback + delay.  
  6. Repeat for other angles and other bottle fill states.  

## Project Structure
```
.
├── audio_segments/ # 2-sec waveform chunks
├── processed_audio/ # Filtered/cleaned signals (optional)
├── raw_videos/ # Original recordings
├── recorded_audio/ # Extracted .wav files
├── recorded_images/ # Extracted .jpg frames
│
├── test_videos/
│ ├── battery/ # Validation clips
│ └── exxp/ # Experimental cases
│
├── notebooks/
│ ├── generate_ultrasonic.ipynb # Create chirp waveform
│ ├── final_training_fusion.ipynb # Train multimodal model
│ └── final_testing_fusion.ipynb # Evaluate performance
│
│
├── README.md
├── requirements.txt

---

## ⚙️ Setup & Installation
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure **ffmpeg** is installed on your system (for audio/video extraction).

3. Run the final_training_fusion.ipynb.

---

## Model Architecture

Audio (Mel Spectrogram) ──► ResNet18 ─┐
                                       ├──► [Concat + FC] ─► Prediction
Image (RGB Frame)       ─► ResNet18 ──┘


## Author

Shamit — research in acoustic & optical signal processing for battery-health detection