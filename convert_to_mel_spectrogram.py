import librosa
import numpy as np
import os

def process_fixed_length_audio(audio_path, sr=22050, n_mels=128, hop_length=512, norm_range=(0, 1)):
    """
    Processes a fixed-length (1-second) audio file into a normalized mel-spectrogram.

    Parameters:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the processed mel-spectrogram as a NumPy array.
        sr (int): Sampling rate. Default is 22050 Hz.
        n_mels (int): Number of mel bands. Default is 128.
        hop_length (int): Number of samples between successive frames. Default is 512.
        norm_range (tuple): Range for normalization (default is (0, 1)).
    """
    # Load the audio file (trim to 1 second)
    y, sr = librosa.load(audio_path, sr=sr, duration=1.0)

    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_min, mel_max = norm_range
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())  # Scale to [0, 1]
    mel_spec_db = mel_spec_db * (mel_max - mel_min) + mel_min  # Scale to desired range

    # # Save as NumPy array
    # np.save(output_path, mel_spec_db)
    # print(f"Processed mel-spectrogram saved to {output_path}")

    return mel_spec_db

# # Example usage
# audio_path = 'path/to/audio.wav'
# output_path = 'path/to/save/mel_spectrogram.npy'

# process_fixed_length_audio(audio_path, output_path)