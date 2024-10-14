import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from einops import rearrange, repeat
import os
import rasterio  # For reading .tif files

def create_dataloaders(csv_file, data_dir, batch_size=64, time_steps=1, scale_factor=0.0001, sequence_length=10,
                       val_ratio=0.2, test_ratio=0.1, patch_size=32,embed_dim=128, image_size=512):
    """
    Function to create train, validation, and test dataloaders.

    Args:
        csv_file (string): Path to the CSV file.
        data_dir (string): Directory for the images.
        batch_size (int): Batch size for the DataLoader.
        time_steps (int): Number of time steps for the sequence.
        scale_factor (float): Scaling factor for NDVI values.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        patch_size (int): Size of patches to extract from images.
        image_size (int): Size to resize images for consistency.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    
    # Create dataset
    dataset = NDVIDatasetAutoencoder(csv_file, data_dir,sequence_length=sequence_length, patch_size=patch_size,embed_dim=128, image_size=image_size, transform=transform)
    
    # Calculate sizes for splits
    total_samples = len(dataset)
    test_size = int(total_samples * test_ratio)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - test_size - val_size

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class NDVIDatasetAutoencoder(Dataset):
    def __init__(self, csv_file, data_dir, sequence_length=10, patch_size=16, embed_dim=128, scale_factor=0.0001, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image file names, dates, and region numbers.
            data_dir (string): Directory where the images are stored.
            sequence_length (int): Number of time steps for each sequence.
            patch_size (int): Size of patches (for ViT, e.g., 16x16).
            embed_dim (int): The dimension to project flattened patches into.
            scale_factor (float): Scaling factor for NDVI values.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.transform = transform
        self.scale_factor = scale_factor

        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)

        # Group by region and sort by date for each region
        self.grouped_data = self.data.groupby('Region Number').apply(
            lambda x: x.sort_values('Date')).reset_index(drop=True)

        # Extract unique regions from the data
        self.regions = self.grouped_data['Region Number'].unique()

        # Create a mapping of region to its image indices
        self.region_indices = {region: self.grouped_data[self.grouped_data['Region Number'] == region].index.tolist() for region in self.regions}

        # Learnable projection layer for patches
        self.projection_layer = torch.nn.Linear(patch_size * patch_size, embed_dim)

    def __len__(self):
        total_sequences = 0
        for region in self.regions:
            total_sequences += max(0, len(self.region_indices[region]) - self.sequence_length)
        return total_sequences

    def split_into_patches(self, image):
        """
        Split the image into non-overlapping patches.
        Args:
            image (ndarray): The input image as a 2D array.

        Returns:
            patches (ndarray): Flattened patches of the image.
        """
        h, w = image.shape
        patches = []

        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                
                # Flatten the patch and append it to the list
                if patch.shape == (self.patch_size, self.patch_size):
                    patches.append(patch.flatten())  # Flatten each patch (e.g., 16x16 -> 256)

        return np.stack(patches)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sequence.

        Returns:
            X (tensor): Input patches for the autoencoder (sequence_length, num_patches, embed_dim).
            Y (tensor): The same as X, since it's an autoencoder.
        """
        # Find the corresponding region and image index for the given idx
        cumulative_idx = 0
        for region in self.regions:
            indices = self.region_indices[region]
            num_sequences = max(0, len(indices) - self.sequence_length)

            if cumulative_idx + num_sequences > idx:
                sequence_start = idx - cumulative_idx
                region_data = self.grouped_data[self.grouped_data['Region Number'] == region]

                # Get the sequence of image paths
                recent_images = region_data.iloc[sequence_start:sequence_start + self.sequence_length]

                X_patches = []
                for _, row in recent_images.iterrows():
                    img_path = os.path.join(self.data_dir, row['File Name'])

                    # Use rasterio to open .tif files
                    with rasterio.open(img_path) as src:
                        image = src.read(1)  # NDVI is usually in the first band
                        image = image * self.scale_factor  # Scale NDVI values

                        if self.transform:
                            image = self.transform(image)

                        # Split image into patches
                        patches = self.split_into_patches(image)

                        # Project flattened patches into embedding space using the learnable projection layer
                        patches = torch.tensor(patches, dtype=torch.float32)
                        patch_embeddings = self.projection_layer(patches)  # Shape: (num_patches, embed_dim)

                        X_patches.append(patch_embeddings)
                
                # Stack the patches to create a sequence (sequence_length, num_patches, embed_dim)
                X_patches = torch.stack(X_patches, dim=0)  # Shape: (sequence_length, num_patches, embed_dim)
                
    
                return X_patches, X_patches

            cumulative_idx += num_sequences

        raise IndexError("Index out of range.")


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