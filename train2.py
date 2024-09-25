from data.patch_data import NDVIDataset
from model.model_2 import NDVIModel
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

sequence_length = 10
num_channels = 100
img_size = (32, 32)
batch_size = 8

cover_model = NDVIModel(sequence_length=sequence_length,num_channels = num_channels,img_size=img_size,lstm_hidden_size=128)
cover_model.to(device)

ndvi_3d = np.load('process_data.npy')
data_cords = np.load('data_cords.npy')


train_dataset = NDVIDataset(ndvi_3d,data_cords,mode="train")
test_dataset = NDVIDataset(ndvi_3d,data_cords,mode="test")
val_dataset = NDVIDataset(ndvi_3d,data_cords,mode="val")

train_dataloader = DataLoader(train_dataset,batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_dataset,batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=8, shuffle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('EPOCHS', type=int, help='number of epochs')
    parser.add_argument('LR', type=float, help='learning rate')
    args = parser.parse_args()

    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001


    history,cover_model = fit(EPOCHS, LR, cover_model, train_dataloader,val_dataloader)
    
    np.save(f'history{datetime.now().strftime("%Y-%m-%d_%H:%M")}.npy', history,allow_pickle=True)
    np.save(f'cover_modelhistory{datetime.now().strftime("%Y-%m-%d_%H:%M")}.npy', cover_model,allow_pickle=True)