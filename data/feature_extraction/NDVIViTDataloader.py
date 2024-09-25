import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import joblib

class NDIVIViTDataloader(Dataset):
    def __init__(self, data,img_size=(64,64),mode='train', scaler=None):

        T, P, H, W = data.shape
        self.img_size = img_size
        train_size = int(0.75 * T * P)
        val_size = int(0.8 * T * P)
        print()
        # Initialize or load the scaler
        self.scaler = StandardScaler() if scaler is None else scaler
      
        if mode == 'train':
            self.ndvi_values = rearrange(data[:train_size],"T P H W -> (T P) (H W)", T=train_size, P=P, H=H, W=W)
            self.ndvi_values = self.scaler.fit_transform(self.ndvi_values)
            self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=tran_size, P=P, H=H, W=W)
        elif mode == "val":
            self.ndvi_values = rearrange(data[train_size:val_size],"T P H W -> (T P) (H W)", T=data[train_size:val_size].shape[0], P=P, H=H, W=W)
            self.ndvi_values = self.scaler.transform(self.ndvi_values)
            self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=data[train_size:val_size].shape[0], P=P, H=H, W=W)
        else:  
            self.ndvi_values = rearrange(data[val_size:],"T P H W -> (T P) (H W)", T=data[val_size:].shape[0], P=P, H=H, W=W)
            self.ndvi_values = self.scaler.transform(self.ndvi_values)
            self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=data[val_size:].shape[0], P=P, H=H, W=W)
    


    def __len__(self):
        return len(self.ndvi_values)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.ndvi_values[idx])
        y = torch.FloatTensor(self.ndvi_values[idx]).unsqueeze(0)
        
        return x,(y,[])
    
    def save_scaler(self, path):
        """Save the scaler to a file."""
        joblib.dump(self.scaler, path)

    @staticmethod
    def load_scaler(path):
        """Load the scaler from a file."""
        return joblib.load(path)
