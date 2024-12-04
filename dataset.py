import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from convert_to_mel_spectrogram import process_fixed_length_audio

class CoughDataset(Dataset):
    def __init__(self, df, audio_dir, sr=22050, n_mels=128, hop_length=512, norm_range=(0, 1), transform=None):
        """
        Dataset for mel-spectrograms from audio files based on a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): A DataFrame containing file names and labels.
            audio_dir (str): Directory containing audio files.
            sr (int): Sampling rate. Default is 22050.
            n_mels (int): Number of mel bands. Default is 128.
            hop_length (int): Number of samples between frames. Default is 512.
            norm_range (tuple): Range for normalization (default is (0, 1)).
            transform (callable): Optional transform to apply to spectrograms.
        """
        self.df = df
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.norm_range = norm_range
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name and label from the DataFrame
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = row['Label']

        # Load audio file
        audio_path = f'{os.path.join(self.audio_dir, file_name)}.wav'
        mel_spec_db = process_fixed_length_audio(audio_path, self.sr, self.n_mels, self.hop_length, self.norm_range)

        # Convert to tensor
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label