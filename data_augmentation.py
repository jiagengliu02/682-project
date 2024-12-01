import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, TimeStretch, TimeMask
import torch
import torchaudio
import os

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr)

def time_warping(audio, rate=1.1):
    return librosa.effects.time_stretch(audio, rate=rate)

def time_masking(audio, sr, mask_duration=0.1):
    augmenter = Compose([TimeMask(min_band_part=0.0, max_band_part=mask_duration)])
    augmented_audio = augmenter(samples=audio, sample_rate=sr)
    return augmented_audio

def frequency_masking(audio, sr, mask_freq=0.1): 
    audio_tensor = torch.tensor(audio).unsqueeze(0) 
    augmented_audio_tensor = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(mask_freq * sr))(audio_tensor) 
    augmented_audio = augmented_audio_tensor.squeeze().numpy() 
    return augmented_audio

def flac_augmentation(file_path):
    output_path = file_path.split(".")[0]
    audio, sr = load_audio(file_path)
    augmented_audio_time_warping = time_warping(audio)
    save_audio(output_path+"-time-warping.flac", augmented_audio_time_warping, sr)
    augmented_audio_time_masking = time_masking(audio, sr)
    save_audio(output_path+"-time-masking.flac", augmented_audio_time_masking, sr)
    augmented_audio_frequency_masking = frequency_masking(audio, sr)
    save_audio(output_path+"-frequency-masking.flac", augmented_audio_frequency_masking, sr)
    return

if __name__ == "__main__":
    base_dir = "/scratch/workspace/yianwang_umass_edu-shared/ljg/682-project/data/LibriSpeech/train-clean-100"
    items = os.listdir(base_dir)
    for item in items:
        item1s = os.listdir(os.path.join(base_dir, item))
        for item1 in item1s:
            item2s = os.listdir(os.path.join(base_dir,item, item1))
            for item2 in item2s:
                file_path = os.path.join(base_dir,item,item1,item2)
                if file_path.endswith("flac"):
                    if file_path.endswith("-time-warping.flac") or file_path.endswith("-time-masking.flac") or file_path.endswith("-frequency-masking.flac"):
                        continue
                    print(file_path)
                    flac_augmentation(file_path=file_path)

