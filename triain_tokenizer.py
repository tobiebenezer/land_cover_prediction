from model.feature_extraction.data_loader import PatchDataset
from model.feature_extraction.unet2d import CNNtokenizer
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

tokenizer = CNNtokenizer(dim = [
    (1, 64),    
    (64, 64),  
    (64, 128),  
    (128, 128), 
    (128, 256), 
    (256, 256), 
    # (256, 512), 
    # (512, 512),
])
tokenizer.to(device)

ndvi_3d = np.load('64x64_patches.npy')

class Scaler():
    def transform(self,x):
        return x/1000
    
    def fit_transform(self,x):
        return x/1000

    def inverse_transform(self,x):
        return x*1000


if not os.path.exists('scaler.pkl'):
    train_dataset = PatchDataset(ndvi_3d,mode="train")
    train_dataset.save_scaler('scaler.pkl')
    scaler = train_dataset.scaler
    
else:
    scaler = PatchDataset.load_scaler('scaler.pkl')
    # scaler = Scaler()
    train_dataset = PatchDataset(ndvi_3d,mode="train",scaler=scaler)

test_dataset = PatchDataset(ndvi_3d,mode="test",scaler=scaler)
val_dataset = PatchDataset(ndvi_3d,mode="val",scaler=scaler)

train_dataloader = DataLoader(train_dataset,batch_size=25, shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset,batch_size=25, shuffle=True,num_workers=2)
test_dataloader = DataLoader(test_dataset,batch_size=25, shuffle=True,num_workers=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('EPOCHS', type=int, help='number of epochs')
    parser.add_argument('LR', type=float, help='learning rate')
    args = parser.parse_args()

    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001

    history,tokenizer = fit(EPOCHS, LR, tokenizer, train_dataloader,val_dataloader)
    
    torch.save(tokenizer.state_dict(), f'cnn_tokenizer_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    tokenizer.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'tokenizerhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)