import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch

# Reshape image into patches of size patch_size
def split_dataset(dataset, val_size=0.15, test_size=0.15):
    torch.manual_seed(101)
    total_size = len(dataset)
    test_len = int(total_size * test_size)
    val_len = int(total_size * val_size)
    train_len = total_size - val_len - test_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    return train_set, val_set, test_set

def get_dataloaders(csv_file, data_dir, NDVIDataset,batch_size=16, patch_size=16, image_size=512,
             transform=None, val_size=0.15, test_size=0.15, shuffle=True,sequence_len=None,pred_len=None):
    # Initialize the full dataset
    if sequence_len is None and pred_len is None:
        dataset = NDVIDataset(csv_file=csv_file, data_dir=data_dir, patch_size=patch_size, image_size=image_size, transform=transform)
    else:
        dataset = NDVIDataset(csv_file=csv_file, data_dir=data_dir, patch_size=patch_size, image_size=image_size,
         transform=transform, x_sequence_length=sequence_len, y_sequence_length=pred_len)


    # Split into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(dataset, val_size=val_size, test_size=test_size)

    # Create DataLoaders for each split
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader