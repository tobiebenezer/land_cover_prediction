import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class NDVIDataset(Dataset):
    def __init__(self, data,cord, sequence_length = 10,img_size=(32,32),mode='train'):

        self.grid_info = np.load('grid_info.npy')
        self.dates =  data.shape[0]
        self.img_size = img_size
        train_size = int(0.75 * self.dates)
        val_size = int(0.8 * self.dates)
        
        self.cord = cord
        
        # Convert to numpy array
        if mode == 'train':
            dim = (*data[:train_size,:,:].shape[:-1],*img_size)
            self.ndvi_values  = data[:train_size,:,:].reshape(dim)
        elif mode == "val":
            dim = (*data[train_size:val_size,:,:].shape[:-1],*img_size)
            self.ndvi_values  = data[train_size:val_size,:,:].reshape(dim)
        else:
            dim = (*data[train_size:,:,:].shape[:-1],*img_size)
            self.ndvi_values  = data[train_size:,:,:].reshape(dim)
            
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.ndvi_values) - self.sequence_length
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.ndvi_values[idx: idx+self.sequence_length])
    
        dim = (self.ndvi_values[idx+self.sequence_length].shape[0],self.img_size[0] * self.img_size[1])    
        y = torch.FloatTensor(self.ndvi_values[idx+self.sequence_length].reshape(dim))
        
        return x,(y,self.cord)