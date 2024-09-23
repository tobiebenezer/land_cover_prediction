import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class NDVIDataset(Dataset):
    def __init__(self, df, sequence_length = 10,img_size=(32,32),mode='train'):
        df_pivot = df.pivot(index=['longitude', 'latitude'], columns='timestamp', values='NDVI')
        df_pivot.fillna(df_pivot.mean(axis=1),inplace=True)
        self.dates =  pd.to_datetime(df_pivot.T.index)
        train_size = int(0.75 * len(self.dates))
        val_size = int(0.8 * len(self.dates))

        # Convert to numpy array
        if mode == 'train':
            self.ndvi_values  = np.stack([ ndv.reshape(img_size) for ndv in df_pivot.T.values[:train_size]])
        elif mode == "val":
            self.ndvi_values  = np.stack([ ndv.reshape(img_size) for ndv in df_pivot.T.values[train_size:val_size]])
        else:
            self.ndvi_values  = np.stack([ ndv.reshape(img_size) for ndv in df_pivot.T.values[train_size:]])
            
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.ndvi_values) - self.sequence_length
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.ndvi_values[idx: idx+self.sequence_length])
        y = torch.FloatTensor(self.ndvi_values[idx+self.sequence_length].reshape( -1))
        
        return x,y