from data.feature_extraction.data_loader import PatchDataset
# from model.feature_extraction.unet2d import CNNtokenizer
from model.TemporalTransformer.tokenizer import NDVIViTFT_tokenizer
from model.reautoencoder import ResNet18Autoencoder
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

# tokenizer = CNNtokenizer(dim = [
#     (1, 64),    
#     (64, 64),  
#     (64, 128),  
#     (128, 128), 
#     (128, 256), 
#     (256, 256), 
#     (256, 512), 
#     (512, 512),
# ])
# tokenizer = NDVIViTFT_tokenizer(depth=2)
tokenizer = ResNet18Autoencoder(in_channels=1, out_channels=1)
tokenizer.to(device)

ndvi_3d = np.load('64x64_patches.npy')

class Scaler():
    def transform(self,x):
        return x/10000
    
    def fit_transform(self,x):
        return x/10000

    def inverse_transform(self,x):
        return x*10000




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--EPOCHS', type=int, help='number of epochs')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, help='batch size')
    parser.add_argument('--NUM_WORKERS', type=int, help='number of workers',default=1)
    args = parser.parse_args()


    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001
    BATCH_SIZE = args.BATCH_SIZE if args.BATCH_SIZE else 2
    NUM_WORKERS = args.NUM_WORKERS if args.NUM_WORKERS else 1

    if not os.path.exists('scaler.pkl'):
        train_dataset = PatchDataset(ndvi_3d,mode="train")
        train_dataset.save_scaler('scaler.pkl')
        scaler = train_dataset.scaler
        
    else:
        # scaler = PatchDataset.load_scaler('scaler.pkl')
        scaler = Scaler()
        train_dataset = PatchDataset(ndvi_3d,mode="train",scaler=scaler)

    test_dataset = PatchDataset(ndvi_3d,mode="test",scaler=scaler)
    val_dataset = PatchDataset(ndvi_3d,mode="val",scaler=scaler)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)


    history,tokenizer = fit(EPOCHS, LR, tokenizer, train_dataloader,val_dataloader)
    
    torch.save(tokenizer.state_dict(), f'cnn_tokenizer_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    tokenizer.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'tokenizerhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)