from data.feature_extraction.NDVIViTDataloader import NDIVIViTDataloader
from model.TemporalTransformer.NDVIViTEncoder import NDVIViTEncoder
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

modelencoder =  NDVIViTEncoder(image_size=64,num_patches=20, patch_size=3, in_channel=1, dim=128, depth=2, heads=8, mlp_ratio=4.)

ndvi_3d = np.load('64x64_patches.npy')

class Scaler():
    def transform(self,x):
        return x/10000
    
    def fit_transform(self,x):
        return x/10000

    def inverse_transform(self,x):
        return x*10000


if not os.path.exists('scaler.pkl'):
    train_dataset = NDIVIViTDataloader(ndvi_3d,mode="train")
    train_dataset.save_scaler('scaler.pkl')
    scaler = train_dataset.scaler
    
else:
    # scaler = NDIVIViTDataloader.load_scaler('scaler.pkl')
    scaler = Scaler()
    train_dataset = NDIVIViTDataloader(ndvi_3d,mode="train",scaler=scaler)

test_dataset = NDIVIViTDataloader(ndvi_3d,mode="test",scaler=scaler)
val_dataset = NDIVIViTDataloader(ndvi_3d,mode="val",scaler=scaler)

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

    history,modelencoder = fit(EPOCHS, LR, modelencoder, train_dataloader,val_dataloader)
    
    torch.save(modelencoder.state_dict(), f'cnn_modelencoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    modelencoder.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'modelencoderhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)