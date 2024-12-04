import pandas as pd
import os
from pydub import AudioSegment
import numpy as np
from extract_energetic_segment import find_most_energetic_segment, save_segment
import librosa
import numpy as np
import soundfile as sf

def reduce_noise_and_extract_cough(input_file, output_file, noise_reduction_level=1.5):
    """
    Remove background noise and extract a 1-second segment of a cough.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to save the extracted cough segment.
    - noise_reduction_level (float): Scale factor for noise reduction, 
      higher values result in stronger noise reduction.
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Estimate noise power using a small segment (e.g., first 0.5 seconds)
    noise_sample = y[:int(0.5 * sr)]
    noise_power = np.mean(np.abs(librosa.stft(noise_sample))**2, axis=1)

    # Perform Short-Time Fourier Transform (STFT) on the audio
    stft = librosa.stft(y)
    magnitude, phase = np.abs(stft), np.angle(stft)
    
    # Create a noise threshold (spectral gating)
    noise_threshold = noise_reduction_level * noise_power[:, np.newaxis]
    
    # Suppress noise below the threshold
    magnitude_denoised = np.maximum(magnitude - noise_threshold, 0)
    
    # Reconstruct the denoised audio using inverse STFT
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(stft_denoised)
    
    # Detect cough using amplitude threshold
    amplitude_threshold = 0.05  # Adjust this based on the audio
    cough_indices = np.where(np.abs(y_denoised) > amplitude_threshold)[0]

    if len(cough_indices) == 0:
        print(input_file)
        raise ValueError("No cough detected in the audio.")
    
    # Extract a 1-second segment around the cough
    start_idx = max(0, cough_indices[0] - sr // 2)  # Start 0.5 seconds before the cough
    end_idx = min(len(y_denoised), start_idx + sr)  # Ensure the segment is 1 second
    cough_segment = y_denoised[start_idx:end_idx]
    
    # Save the cough segment
    sf.write(output_file, cough_segment, sr)
    print(f"Cough segment saved to {output_file}")

if __name__ == '__main__':

    path_to_data_folder = '/home/sameer/Cough-Sense/data'
    file_name = 'dataset_labels.csv'

    df = pd.read_csv(f'{path_to_data_folder}/{file_name}')
    filtered_df = df.loc[(df['Label'] == 'viral') | (df['Label'] == 'neither')]

    covid_file_list = filtered_df['file_name'].tolist()

    for file in covid_file_list:

        input_file = f"{path_to_data_folder}/{file}.wav"
        output_file = input_file
        try:
            reduce_noise_and_extract_cough(input_file, output_file, noise_reduction_level=1.5)
        except Exception as e:
            continue