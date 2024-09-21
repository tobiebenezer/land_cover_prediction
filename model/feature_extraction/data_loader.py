import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class PatchDataset(Dataset):
    def __init__(self, data,img_size=(64,64),mode='train'):

        T, P, H, W = data.shape
        reshaped_data = data.reshape(T * P, H, W)
        self.img_size = img_size
        train_size = int(0.75 * self.dates)
        val_size = int(0.8 * self.dates)

        # Convert to numpy array
        if mode == 'train':
            self.ndvi_values  = reshaped_data[:train_size]
        elif mode == "val":
            self.ndvi_values  = reshaped_data[train_size:val_size]
        else:
            self.ndvi_values  = data[train_size:]
        
    def __len__(self):
        return len(self.ndvi_values)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.ndvi_values[idx])
        y = torch.FloatTensor(self.ndvi_values[idx])
        
        return x,(y,[])