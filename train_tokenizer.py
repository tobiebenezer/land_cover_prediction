from data.feature_extraction.data_loader import PatchDataset
from data.feature_extraction.autoencoder_dataloader import NDVIDataset
from model.feature_extraction.unet2d import CNNtokenizer
from model.TemporalTransformer.tokenizer import NDVIViTFT_tokenizer
from model.reautoencoder import ResNet18Autoencoder
from model.feature_extraction.conv_autoencoder import CAE
from utils.traning import *
from utils.process_data import get_data
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from data.feature_extraction.NDVIViTDataloader import NDIVIViTDataloader
import argparse
from utils.dataloader import get_dataloaders



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# tokenizer = CNNtokenizer()
# tokenizer = NDVIViTFT_tokenizer(depth=2)

Tokenizer = {
    'CNNtokenizer': CNNtokenizer(),
    'custom_CNNtokenizer': CNNtokenizer(dim = [
                                            (1, 64),    
                                            (64, 64),  
                                            (64, 128),  
                                            (128, 128), 
                                            (128, 256), 
                                            (256, 256), 
                                            (256, 512), 
                                            (512, 512),
                                        ]),
    
    'NDVIViTFT_tokenizer': NDVIViTFT_tokenizer(depth=2),
    'ResNet18Autoencoder': ResNet18Autoencoder(in_channels=1, out_channels=1),
    'CAE': CAE()
}






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training tokenizer \n MODEL_NAME: CNNtokenizer, custom_CNNtokenizer, NDVIViTFT_tokenizer, ResNet18Autoencoder')
    parser.add_argument('--EPOCHS', type=int, help='number of epochs')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, help='batch size')
    parser.add_argument('--NUM_WORKERS', type=int, help='number of workers',default=1)
    parser.add_argument('--IMG_LOG', type=str, help='image log',default='processed_images_log.csv')
    parser.add_argument('--DATA_DIR', type=str, help='data directory',default='extracted_data')
    parser.add_argument('--PATCH_SIZE', type=int, help='patch size',default=64)
    parser.add_argument('--IMAGE_SIZE', type=int, help='image size',default=512)
    parser.add_argument('--VAL_SIZE', type=float, help='validation size',default=0.15)
    parser.add_argument('--TEST_SIZE', type=float, help='test size',default=0.15)
    parser.add_argument('--MODEL_NAME', type=str, help='model name',default='cnn_tokenizer')
    parser.add_argument('--ACCUMULATION_STEPS', type=int, help='accumulation steps',default=3)


    args = parser.parse_args()


    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001
    BATCH_SIZE = args.BATCH_SIZE if args.BATCH_SIZE else 2
    NUM_WORKERS = args.NUM_WORKERS if args.NUM_WORKERS else 1
    IMG_LOG = args.IMG_LOG if args.IMG_LOG else 'processed_images_log.csv'
    DATA_DIR = args.DATA_DIR if args.DATA_DIR else 'extracted_data'
    PATCH_SIZE = args.PATCH_SIZE if args.PATCH_SIZE else 64
    IMAGE_SIZE = args.IMAGE_SIZE if args.IMAGE_SIZE else 512
    VAL_SIZE = args.VAL_SIZE if args.VAL_SIZE else 0.15
    TEST_SIZE = args.TEST_SIZE if args.TEST_SIZE else 0.15
    MODEL_NAME = args.MODEL_NAME if args.MODEL_NAME else 'cnn_tokenizer'
    ACCUMULATION_STEPS = args.ACCUMULATION_STEPS if args.ACCUMULATION_STEPS else 3
    

    # Load the data
    csv_file = IMG_LOG
    data_dir = DATA_DIR
    patch_size = PATCH_SIZE
    image_size = IMAGE_SIZE
    batch_size = BATCH_SIZE
    val_size = VAL_SIZE
    test_size = TEST_SIZE

    tokenizer = Tokenizer[MODEL_NAME].to(device)

    optimizer = optim.Adam
    # Create the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(csv_file, data_dir, NDVIDataset, 
            batch_size=batch_size, patch_size=patch_size, image_size=image_size, val_size=val_size, test_size=test_size)
      
    history,tokenizer = fit(EPOCHS, LR, tokenizer, train_dataloader,val_dataloader, optimizer,accumulation_steps=ACCUMULATION_STEPS)
    
    torch.save(tokenizer.state_dict(), f'{MODEL_NAME}_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    tokenizer.save_weights(f'{MODEL_NAME}_encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'{MODEL_NAME}_decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')




# ndvi_3d = np.load('64x64_patches.npy')

# class Scaler():
#     def transform(self,x):
#         return x/10000
    
#     def fit_transform(self,x):
#         return x/10000

#     def inverse_transform(self,x):
#         return x*10000
    np.save(f'{MODEL_NAME}_tokenizerhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)