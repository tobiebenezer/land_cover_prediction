import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from einops import rearrange, repeat

class NDVIDataset(Dataset):
    def __init__(self, data, cord, sequence_length=10, img_size=(32, 32), mode='train'):
        self.dates = data.shape[0]
        self.img_size = img_size
        train_size = int(0.75 * self.dates)
        val_size = int(0.8 * self.dates)
        
        self.cord = cord
        
        # Convert to numpy array
        if mode == 'train':
            self.ndvi_values = data[:train_size]
        elif mode == "val":
            self.ndvi_values = data[train_size:val_size]
        else:
            self.ndvi_values = data[val_size:]
        
        self.sequence_length = sequence_length
        
        # Reshape the data to (patches, time, height, width)
        self.ndvi_values = rearrange(self.ndvi_values, 't (h p1) (w p2) -> (h w) t p1 p2', 
                                     p1=img_size[0], p2=img_size[1])
        
        # Number of patches
        self.num_patches = self.ndvi_values.shape[0]
        
    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        # Get the sequence for a single patch
        patch_sequence = self.ndvi_values[idx]
        
        # Split into input and target
        x = patch_sequence[:-1]  # All but the last time step
        y = patch_sequence[-1]   # The last time step
        
        # Stack the last time step on itself to match the input shape
        y_stacked = np.stack([y] * (self.sequence_length - 1))
        
        return torch.FloatTensor(x), torch.FloatTensor(y_stacked)
# class NDVIDataset(Dataset):
#     def __init__(self, data, cord, sequence_length=10, forecast_horizon=5, img_size=(32, 32), mode='train'):
#         self.dates = data.shape[0]
#         self.img_size = img_size
#         train_size = int(0.75 * self.dates)
#         val_size = int(0.8 * self.dates)
        
#         self.cord = cord
        
#         # Convert to numpy array
#         if mode == 'train':
#             self.ndvi_values = data[:train_size]
#         elif mode == "val":
#             self.ndvi_values = data[train_size:val_size]
#         else:
#             self.ndvi_values = data[val_size:]
        
#         self.sequence_length = sequence_length
#         self.forecast_horizon = forecast_horizon
        
#         # Reshape and transpose the data using einops
#         self.ndvi_values = rearrange(self.ndvi_values, 't p (h1 w1) -> p t h1 w1', h1=img_size[0], w1=img_size[1])
        
#     def __len__(self):
#         return self.ndvi_values.shape[1] - self.sequence_length - self.forecast_horizon + 1
    
#     def __getitem__(self, idx):
#         x = rearrange(self.ndvi_values[:, idx:idx+self.sequence_length], 'p t h w -> t p h w')
#         y = rearrange(self.ndvi_values[:, idx+self.sequence_length:idx+self.sequence_length+self.forecast_horizon], 
#                       'p t h w -> t p h w')
        
#         return torch.FloatTensor(x), (torch.FloatTensor(y), self.cord)
# class NDVIDataset(Dataset):
#     def __init__(self, data,cord, sequence_length = 10,img_size=(32,32),mode='train'):

#         self.dates =  data.shape[0]
#         self.img_size = img_size
#         train_size = int(0.75 * self.dates)
#         val_size = int(0.8 * self.dates)
        
#         self.cord = cord
        
#         # Convert to numpy array
#         if mode == 'train':
#             dim = (*data[:train_size,:,:].shape[:-1],*img_size)
#             self.ndvi_values  = data[:train_size,:,:].reshape(dim)
#         elif mode == "val":
#             dim = (*data[train_size:val_size,:,:].shape[:-1],*img_size)
#             self.ndvi_values  = data[train_size:val_size,:,:].reshape(dim)
#         else:
#             dim = (*data[train_size:,:,:].shape[:-1],*img_size)
#             self.ndvi_values  = data[train_size:,:,:].reshape(dim)
            
#         self.sequence_length = sequence_length
        
#     def __len__(self):
#         return len(self.ndvi_values) - self.sequence_length
    
#     def __getitem__(self,idx):
#         x = torch.FloatTensor(self.ndvi_values[idx: idx+self.sequence_length])
    
#         dim = (self.ndvi_values[idx+self.sequence_length].shape[0],self.img_size[0] * self.img_size[1])    
#         y = torch.FloatTensor(self.ndvi_values[idx+self.sequence_length].reshape(dim))
        
#         return x,(y,self.cord)