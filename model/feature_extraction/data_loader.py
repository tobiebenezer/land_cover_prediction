import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class PatchDataset(Dataset):
    def __init__(self, data,img_size=(64,64),mode='train'):

        T, P, H, W = data.shape
        reshaped_data = data.reshape(T * P, H, W)
        self.img_size = img_size
        train_size = int(0.75 * T * P)
        val_size = int(0.8 * T * P)

        # Convert to numpy array
        if mode == 'train':
            self.ndvi_values  = reshaped_data[:train_size].reshape(train_size, *self.img_size)
        elif mode == "val":
            self.ndvi_values  = reshaped_data[train_size:val_size].reshape(val_size - train_size, *self.img_size)
        else:
            self.ndvi_values  = reshaped_data[train_size + val_size:].reshape(((T * P )- (train_size+val_size)), *self.img_size)
        
    def __len__(self):
        return len(self.ndvi_values)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.ndvi_values[idx])
        y = torch.FloatTensor(self.ndvi_values[idx]).unsqueeze(1)
        
        return x,(y,[])