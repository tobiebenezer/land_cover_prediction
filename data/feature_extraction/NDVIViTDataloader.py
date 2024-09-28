import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import joblib

# class NDIVIViTDataloader(Dataset):
#     def __init__(self, data,context,sequence_length = 16, pred_size = 4, img_size=(64,64),mode='train', scaler=None):

#         T, P, H, W = data.shape
#         self.img_size = img_size
#         train_size = int(0.75 * T )
#         val_size = int(0.8 * T )
#         self.sequence_length = sequence_length
#         self.pred_size = pred_size

#         # Initialize or load the scaler
#         self.scaler = StandardScaler() if scaler is None else scaler
      
#         if mode == 'train':
#             self.ndvi_values = rearrange(data[:train_size],"T P H W -> (T P) (H W)", T=train_size, P=P, H=H, W=W)
#             self.ndvi_values = self.scaler.fit_transform(self.ndvi_values)
#             self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=train_size, P=P, H=H, W=W)
#             self.context = context[:train_size]
                        
#         elif mode == "val":
#             self.ndvi_values = rearrange(data[train_size:val_size],"T P H W -> (T P) (H W)", T=data[train_size:val_size].shape[0], P=P, H=H, W=W)
#             self.ndvi_values = self.scaler.transform(self.ndvi_values)
#             self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=data[train_size:val_size].shape[0], P=P, H=H, W=W)
#             self.context = context[train_size:val_size]

#         else:  
#             self.ndvi_values = rearrange(data[val_size:],"T P H W -> (T P) (H W)", T=data[val_size:].shape[0], P=P, H=H, W=W)
#             self.ndvi_values = self.scaler.transform(self.ndvi_values)
#             self.ndvi_values = rearrange(self.ndvi_values,"(T P) (H W) -> T P H W", T=data[val_size:].shape[0], P=P, H=H, W=W)
#             self.context = context[val_size:]
    


#     def __len__(self):
#         return len(self.ndvi_values)
    
#     def __getitem__(self, idx):
#         x = torch.FloatTensor(self.ndvi_values[idx: idx+self.sequence_length - self.pred_size]).contiguous()
#         y = torch.FloatTensor(self.ndvi_values[idx+self.sequence_length - self.pred_size : idx+self.sequence_length]).unsqueeze(1).contiguous()
#         context = torch.FloatTensor(self.context[idx: idx+self.sequence_length - self.pred_size]).contiguous()
        
#         # Ensure consistent shapes
#         # x = x.view(-1, *self.img_size)
#         # y = y.view(-1, *self.img_size)
        
#         return x, context, (y, [])
    
#     def save_scaler(self, path):
#         """Save the scaler to a file."""
#         joblib.dump(self.scaler, path)

#     @staticmethod
#     def load_scaler(path):
#         """Load the scaler from a file."""
#         return joblib.load(path)


class NDIVIViTDataloader(Dataset):
    def __init__(self, data, context, sequence_length=16, pred_size=4, img_size=(64,64), mode='train', scaler=None):
        T, P, H, W = data.shape
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.pred_size = pred_size

        train_size = int(0.75 * T)
        val_size = int(0.8 * T)

        # Initialize or load the scaler
        self.scaler = StandardScaler() if scaler is None else scaler

        if mode == 'train':
            self.ndvi_values = data[:train_size]
            self.context = context[:train_size]
        elif mode == "val":
            self.ndvi_values = data[train_size:val_size]
            self.context = context[train_size:val_size]
        else:  # test
            self.ndvi_values = data[val_size:]
            self.context = context[val_size:]

        # Apply scaling
        self.ndvi_values = self._scale_data(self.ndvi_values)

        # Ensure that we have enough data for at least one complete sequence
        self.valid_indices = list(range(len(self.ndvi_values) - self.sequence_length + 1))

    def _scale_data(self, data):
        T, P, H, W = data.shape
        flat_data = rearrange(data, "T P H W -> (T P) (H W)")
        if self.scaler.n_samples_seen_ is None:  # If scaler hasn't been fit yet
            scaled_data = self.scaler.fit_transform(flat_data)
        else:
            scaled_data = self.scaler.transform(flat_data)
        return rearrange(scaled_data, "(T P) (H W) -> T P H W", T=T, P=P, H=H, W=W)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = torch.FloatTensor(self.ndvi_values[real_idx:real_idx+self.sequence_length-self.pred_size]).contiguous()
        y = torch.FloatTensor(self.ndvi_values[real_idx+self.sequence_length-self.pred_size:real_idx+self.sequence_length]).unsqueeze(1).contiguous()
        context = torch.FloatTensor(self.context[real_idx:real_idx+self.sequence_length-self.pred_size]).contiguous()

        # Ensure consistent shapes
        x = x.view(-1, 1, *self.img_size)
        y = y.view(-1, 1, *self.img_size)
        context = context.view(-1, context.size(-1))

        return x, context, (y, [])

    def save_scaler(self, path):
        """Save the scaler to a file."""
        joblib.dump(self.scaler, path)

    @staticmethod
    def load_scaler(path):
        """Load the scaler from a file."""
        return joblib.load(path)