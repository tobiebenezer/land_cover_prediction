class NDVIDataset(Dataset):
    def __init__(self, csv_file, data_dir, patch_size=16, image_size=512, transform=None):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.image_size = image_size
        self.transform = transform
        self.scale_factor = 0.0001

        # Load the CSV file
        self.data = pd.read_csv(csv_file)

        # Ensure that 'Region' and 'Date' columns exist in the CSV
        if 'Region Number' not in self.data.columns or 'Date' not in self.data.columns:
            raise ValueError("CSV must contain 'Region' and 'Date' columns.")

        # Sort by 'Region' first, then by 'Date' within each region
        self.data = self.data.sort_values(by=['Region Number', 'Date'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image file path from the CSV
        img_path = os.path.join(self.data_dir, self.data.iloc[idx]['File Name'])

        # Load the image using rasterio and apply scaling
        with rasterio.open(img_path) as src:
            image = src.read(1).astype(np.float32) * self.scale_factor

        # Apply any transforms if provided (e.g., augmentations)
        if self.transform:
            image = self.transform(image)

        # Ensure the image is resized to 512x512
        image = F.interpolate(torch.tensor(image).unsqueeze(0).unsqueeze(0),
                              size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False).squeeze()

        # Reshape image into patches of size patch_size x patch_size
        patches = rearrange(image, '(h p1) (w p2) -> (h w) 1 p1 p2',
                            p1=self.patch_size, p2=self.patch_size)

        # Return the patches (as inputs) and the original full image (as target)
        return patches, image


