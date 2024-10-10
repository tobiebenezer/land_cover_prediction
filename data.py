from utils.data_downloader import NDVIGEE_Extractor, Process_TifDATA
import argparse
import pandas as pd
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data downloader')
    parser.add_argument('--START_DATE', type=int, help='start date')
    parser.add_argument('--END_DATE', type=int, help='end date')
    parser.add_argument('--OUTPUT_FILE', type=str, help='output file')  
    parser.add_argument('--EXTRACTED_DIR', type=str, help='extracted directory')
    parser.add_argument('--ZIP_DIR', type=str, help='zip directory')
    parser.add_argument('--LOG_FILE', type=str, help='log file')
    parser.add_argument('--IMG_COLLECTION', type=str, help='image collection')
    
    
    args = parser.parse_args()

    START_DATE = args.START_DATE if args.START_DATE else 2019
    END_DATE = args.END_DATE if args.END_DATE else 2019
    OUTPUT_FILE = args.OUTPUT_FILE if args.OUTPUT_FILE else 'download_urls.csv'
    EXTRACTED_DIR = args.EXTRACTED_DIR if args.EXTRACTED_DIR else 'extracted_data'
    ZIP_DIR = args.ZIP_DIR if args.ZIP_DIR else 'zip_data'
    LOG_FILE = args.LOG_FILE if args.LOG_FILE else 'processed_images_log.csv'
    IMG_COLLECTION = args.IMG_COLLECTION if args.IMG_COLLECTION else 'MODIS/061/MOD13A1'

    # Download NDVI data from Google Earth Engine
    os.makedirs(ZIP_DIR, exist_ok=True)
    ndvi_extractor = NDVIGEE_Extractor(START_DATE, END_DATE, OUTPUT_FILE, ZIP_DIR, IMG_COLLECTION)
    ndvi_extractor()
    
    # Process downloaded GeoTIFF 
    process_tif = Process_TifDATA(EXTRACTED_DIR, ZIP_DIR, LOG_FILE)

    process_tif()
    print("All images processed and saved to disk.")

