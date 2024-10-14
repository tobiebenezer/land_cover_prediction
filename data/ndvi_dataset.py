import pandas as pd
import numpy as np
import rasterio
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class NDVIDataset(Dataset):
    def __init__(self, csv_file, data_dir, patch_size=16, image_size=512,
                 x_sequence_length=5, y_sequence_length=1, transform=None):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.image_size = image_size
        self.x_sequence_length = x_sequence_length
        self.y_sequence_length = y_sequence_length
        self.sequence_length = x_sequence_length + y_sequence_length
        self.transform = transform
        self.scale_factor = 0.0001

        self.data = pd.read_csv(csv_file)

        if 'Region Number' not in self.data.columns or 'Date' not in self.data.columns:
            raise ValueError("CSV must contain 'Region Number' and 'Date' columns.")
        self.data = self.data.sort_values(by=['Region Number', 'Date'])
        self.region_groups = self.data.groupby('Region Number')

        # Precompute possible slices (X, Y) per region
        self.slices = []
        for region_number, region_data in self.region_groups:
            for start_idx in range(len(region_data) - self.sequence_length + 1):
                self.slices.append((region_number, start_idx))

    def __len__(self):
        # The length is the total number of slices across all regions
        return len(self.slices)

    def __getitem__(self, idx):

        region_number, start_idx = self.slices[idx]
        region_data = self.region_groups.get_group(region_number)
        x_sequence_data = region_data.iloc[start_idx:start_idx + self.x_sequence_length]
        y_sequence_data = region_data.iloc[start_idx + self.x_sequence_length:
                                           start_idx + self.sequence_length]

        x_image_sequence = []
        y_image_sequence = []
        for _, row in x_sequence_data.iterrows():
            img_path = os.path.join(self.data_dir, row['File Name'])
            with rasterio.open(img_path) as src:
                image = src.read(1).astype(np.float32) * self.scale_factor
            if self.transform:
                image = self.transform(image)
            image = F.interpolate(torch.tensor(image).unsqueeze(0).unsqueeze(0),
                                  size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False).squeeze()
            x_image_sequence.append(image)

        for _, row in y_sequence_data.iterrows():
            img_path = os.path.join(self.data_dir, row['File Name'])
            with rasterio.open(img_path) as src:
                image = src.read(1).astype(np.float32) * self.scale_factor
            if self.transform:
                image = self.transform(image)
            image = F.interpolate(torch.tensor(image).unsqueeze(0).unsqueeze(0),
                                  size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False).squeeze()
            y_image_sequence.append(image)

        # Convert the sequences to tensors
        x_image_sequence = torch.stack(x_image_sequence)
        y_image_sequence = torch.stack(y_image_sequence)
        x_patches_sequence = rearrange(x_image_sequence, 't (h p1) (w p2) -> t (h w) 1 p1 p2',
                                       p1=self.patch_size, p2=self.patch_size)
        y_patches_sequence = rearrange(y_image_sequence, 't (h p1) (w p2) -> t (h w) 1 p1 p2',
                                       p1=self.patch_size, p2=self.patch_size)

        return x_patches_sequence, y_patches_sequence
