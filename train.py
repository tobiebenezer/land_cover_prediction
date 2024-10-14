from data.feature_extraction.data_loader import PatchDataset, NDVIDataset, NDVIDatasetAutoencoder, create_dataloaders
import torch
from model.TemporalTransformer.tokenizer import NDVIViTFT_tokenizer
from model.reautoencoder import ResNet18Autoencoder
from model.vit_autoencoder import ViTAutoencoder
from utils.traning import *
from utils.process_data import get_data
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import argparse


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--EPOCHS', type=int, help='number of epochs')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, help='batch size')
    parser.add_argument('--IMAGES_LOG', type=str, help='path to images log',default='processed_images_log.csv')
    parser.add_argument('--DATA_DIR', type=str, help='path to data directory',default='extracted_data')
    parser.add_argument('--NUM_WORKERS', type=int, help='number of workers',default=1)
    parser.add_argument('--IMAGE_SIZE', type=int, help='image size',default=512)
    parser.add_argument('--PATCH_SIZE', type=int, help='patch size',default=32)
    args = parser.parse_args()


    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001
    BATCH_SIZE = args.BATCH_SIZE if args.BATCH_SIZE else 2
    NUM_WORKERS = args.NUM_WORKERS if args.NUM_WORKERS else 1
    IMAGES_LOG = args.IMAGES_LOG if args.IMAGES_LOG else 'processed_images_log.csv'
    DATA_DIR = args.DATA_DIR if args.DATA_DIR else 'extracted_data'
    IMAGE_SIZE = args.IMAGE_SIZE if args.IMAGE_SIZE else 512
    PATCH_SIZE = args.PATCH_SIZE if args.PATCH_SIZE else 32


    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(csv_file=IMAGES_LOG, data_dir=DATA_DIR, batch_size=BATCH_SIZE, time_steps=1, scale_factor=0.0001, 
                       val_ratio=0.2, test_ratio=0.1, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)

    num_patches = IMAGE_SIZE // PATCH_SIZE

    tokenizer = ViTAutoencoder(embed_dim=128, num_patches=num_patches, sequence_length=10, num_heads=4, transformer_layers=4, mlp_dim=256)
    tokenizer.to(device)

    history,tokenizer = fit(EPOCHS, LR, tokenizer, train_dataloader,val_dataloader)
    
    torch.save(tokenizer.state_dict(), f'cnn_tokenizer_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    tokenizer.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'tokenizerhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)