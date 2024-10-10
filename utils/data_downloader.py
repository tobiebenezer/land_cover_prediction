import os
import zipfile
import rasterio
import ee
import csv
import requests
from pathlib import Path
from skimage.transform import resize
from rasterio.enums import Resampling
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from rasterio.plot import show
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

ee.Initialize()

class NDVIGEE_Extractor:
    def __init__(self, start, end, output_file, img_collection='MODIS/061/MOD13A1'):
        self.start = start
        self.end = end
        self.output_file = output_file
        self.img_collection = img_collection
        
        # Define the area of interest (Zambia)
        self.aoi = ee.Geometry.Polygon(
            [[[21.9999, -17.9625],
              [33.7060, -17.9625],
              [33.7060, -8.2712],
              [21.9999, -8.2712]]]
        )

    def generate_16_day_intervals(self, year):
        start_date = datetime.date(year, 1, 1)
        intervals = []
        while start_date.year == year:
            end_date = start_date + datetime.timedelta(days=15)
            intervals.append((start_date, end_date))
            start_date = end_date + datetime.timedelta(days=1)
        return intervals

    # split the AOI into smaller chunks (4x4 grid)
    def split_aoi(self, grid_size=4):
        coords = self.aoi.bounds().coordinates().getInfo()[0]
        xmin, ymin = coords[0]
        xmax, ymax = coords[2]

        x_step = (xmax - xmin) / grid_size
        y_step = (ymax - ymin) / grid_size

        sub_regions = []
        for i in range(grid_size):
            for j in range(grid_size):
                sub_xmin = xmin + i * x_step
                sub_xmax = xmin + (i + 1) * x_step
                sub_ymin = ymin + j * y_step
                sub_ymax = ymin + (j + 1) * y_step
                sub_region = ee.Geometry.Rectangle([sub_xmin, sub_ymin, sub_xmax, sub_ymax])
                sub_regions.append(sub_region)

        return sub_regions

    # Download each image
    def download_image(self, image, start_date, end_date, count, sub_region, sub_count, writer):
        try:
            url = image.getDownloadURL({
                'scale': 500,  # MODIS resolution is 500m
                'region': sub_region.getInfo()['coordinates'],
                'fileFormat': 'GeoTIFF',
            })

            writer.writerow([count, sub_count, start_date, end_date, url])

        except ee.EEException as e:
            print(f"Failed to download image for sub-region {sub_count}: {e}")

    def write_url_img_year(self, year):
        date_ranges = self.generate_16_day_intervals(year)

        # Split Zambia into smaller sub-regions
        sub_regions = self.split_aoi()

        # Open the CSV file for writing
        with open(self.output_file, mode= 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Image Number', 'Sub-region', 'Start Date', 'End Date', 'Download URL'])

            # Process images for each 16-day interval and each sub-region
            for i, (start, end) in enumerate(date_ranges):
                modis_dataset = ee.ImageCollection(self.img_collection) \
                                  .filterBounds(self.aoi) \
                                  .filterDate(str(start), str(end)) \
                                  .select('NDVI')

                image = modis_dataset.first()

                if not image.getInfo():  # If no image is returned
                    print(f"No image found for date range: {start} to {end}")
                    continue  

                for j, sub_region in enumerate(sub_regions):
                    self.download_image(image, start, end, i + 1, sub_region, j + 1, writer)

        print(f"URLs written to '{self.output_file}'")
        
    def __call__(self):
        years = range(self.start, self.end + 1)
        for year in tqdm(years, desc="Extracting year"):
            self.write_url_img_year(year)


class Process_TifDATA:
    
    palette = ['#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
       '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
       '#012e01', '#011d01', '#011301']
    
    def __init__(self, extracted_dir, zip_dir, log_file):
        self.extracted_dir = extracted_dir
        self.zip_dir = zip_dir
        self.log_file = log_file
        
        os.makedirs(extracted_dir, exist_ok=True)
        
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['File Name', 'Date', 'Region Number'])
                
    def download_image(self,url, image_number, sub_region, start_date, end_date):
        try:
            # Send a GET request to download the image
            response = requests.get(url, stream=True)
            response.raise_for_status() 
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition:
                filename = re.findall("filename=(.+)", content_disposition)[0].strip('"')
            else:
                # If Content-Disposition is not available, extract filename from URL
                filename = unquote(url.split('/')[-1])

            # Extract the date from the filename
            start_date = filename.split('.')[0]
            # Define the file name
            file_name = f"NDVI_Image_{image_number}_Sub_{sub_region}_{start_date}_to_{end_date}.zip"
            file_path = os.path.join(output_dir, file_name)

            # Save the image to disk
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded and saved: {file_name}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

    def extract_zip(self, zip_file_path, extract_to):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_name = Path(zip_file_path).stem
                extract_to_path = Path(extract_to)

                for member in zip_ref.infolist():
                    if member.is_dir():
                        continue  

                    target_path = extract_to_path / member.filename
                    base_name = target_path.stem
                    extension = target_path.suffix

                    counter = 1
                    while target_path.exists():
                        new_name = f"{zip_name}_{base_name}_{counter}{extension}"
                        target_path = extract_to_path / new_name
                        counter += 1

                    # Extract and rename in one step
                    zip_ref.extract(member, extract_to_path)
                    extracted_path = extract_to_path / member.filename
                    extracted_path.rename(target_path)

                    print(f"Extracted: {target_path}")

            print(f"Successfully extracted all files from {zip_file_path}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file_path} is not a valid ZIP file or is corrupted.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def resample_geotiff(self, input_path, output_path, target_resolution=500):
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, 
                src.bounds.left, src.bounds.bottom, 
                src.bounds.right, src.bounds.top, 
                resolution=target_resolution
            )

            kwargs = src.meta.copy()
            kwargs.update({
                'transform': transform,
                'width': width,
                'height': height
            })
            print(f"Resampling to {output_path}")
            
            # Writing the resampled image
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest  
                    )

    def resize_geotiff(self,input_path, output_path, target_shape=(512, 512)):
        with rasterio.open(input_path) as src:
            # Calculate scaling factors
            scale_factor_x = src.width / target_shape[1]
            scale_factor_y = src.height / target_shape[0]

            # Create a new transformation matrix
            transform = src.transform * src.transform.scale(
                scale_factor_x, scale_factor_y)

            # Resample data to target shape
            data = src.read(
                out_shape=target_shape,
                resampling=Resampling.bilinear
            )

            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'driver': 'GTiff',
                'height': target_shape[0],
                'width': target_shape[1],
                'transform': transform
            })

            # Write the resized image
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(data)

    def log_image_data(self, file_name, date, region_number):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([file_name, date, region_number])  
                  
    def _unzip_extract_file(self):
        zip_files = [f for f in os.listdir(self.zip_dir) if f.endswith('.zip')]

        for zip_file in zip_files:
            zip_file_path = os.path.join(self.zip_dir, zip_file)
            self.extract_zip(zip_file_path, self.extracted_dir)
            
    def view_geotiff(self,file_path):
        cmap = LinearSegmentedColormap.from_list('custom_ndvi', self.palette, N=len(self.palette))
        try:

            with rasterio.open(file_path) as dataset:
                ndvi_data = dataset.read(1)  #reading one band
                print(ndvi_data.shape)
                # Plot the NDVI data
                plt.imshow(ndvi_data * 0.0001, cmap=cmap, vmin=-1, vmax=1)
                plt.colorbar(label='NDVI')
                plt.title(f"NDVI Image: {os.path.basename(file_path)}")

        except rasterio.errors.RasterioIOError:
            print(f"Error: {file_path} is not a valid GeoTIFF or is corrupted.")
    
    def __call__(self):
        self._unzip_extract_file()
        
        tif_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.extracted_dir) for file in files if file.endswith('.tif')]
        if tif_files:
            for tif_file in tif_files:
                file_name = os.path.basename(tif_file)
                parts = file_name.split('_')
                if len(parts) >= 5:
                    image_number = parts[2]  
                    region_number = parts[4] 
                    date = parts[5] 
                    
                    # Output file paths for resampled and resized images
                    resized_output_path = os.path.join(self.extracted_dir, f"{file_name}")

                    # Resize the image to 512x512
                    self.resize_geotiff(tif_file, resized_output_path, target_shape=(512, 512))

                    # Log the file details
                    self.log_image_data(file_name, date, region_number)

        else:
            print("No GeoTIFF files found after extraction.")

