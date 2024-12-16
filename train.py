import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import CoughDataset
from model import get_resnet18_model
from utils import train_model, evaluate_model
from sklearn.model_selection import train_test_split

path_to_labels = '/home/sameer/Cough-Sense/data/dataset_labels.csv'

df = pd.read_csv(path_to_labels)
mapping = {"neither": 0, "viral": 1, "bacterial": 2}

# Apply the mapping to the entire DataFrame
df = df.replace(mapping)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Split data into train (80%), validation (10%), and test sets (10%)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

# Audio directory
audio_dir = '/home/sameer/Cough-Sense/data'

# Create datasets
train_dataset = CoughDataset(train_df, audio_dir=audio_dir)
val_dataset = CoughDataset(val_df, audio_dir=audio_dir)
test_dataset = CoughDataset(test_df, audio_dir=audio_dir)

# Initialize hyperparameters
batch_size = 16
num_classes = 3
num_epochs = 10
learning_rate = 0.001

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_resnet18_model(num_classes)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
path_to_model_save = '/home/sameer/Cough-Sense/saved_models/model1.pt'

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, path_to_model_save)

# Evaluate on the test set
evaluate_model(trained_model, test_loader, device, num_classes, path_to_model_save)