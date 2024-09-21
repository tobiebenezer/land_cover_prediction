from model.feature_extraction.data_loader import PatchDataset
from model.feature_extraction.unet2d import CNNtokenizer
from utils.traning import *
from utils.process_data import get_data
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = CNNtokenizer()
tokenizer.to(device)

ndvi_3d = np.load('64x64_patches.npy')

train_dataset = PatchDataset(ndvi_3d,mode="train")
test_dataset = PatchDataset(ndvi_3d,mode="test")
val_dataset = PatchDataset(ndvi_3d,mode="val")

train_dataloader = DataLoader(train_dataset,batch_size=25, shuffle=False)
val_dataloader = DataLoader(val_dataset,batch_size=25, shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=25, shuffle=False)

if __name__ == "__main__":
    EPOCHS = 50
    LR = 0.001

    history,tokenizer = fit(EPOCHS, LR, tokenizer, train_dataloader,val_dataloader)
    
    torch.save(model.state_dict(), f'cnn_tokenizer_weights{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pth')
    tokenizer.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pth')
    np.save(f'tokenizerhistory{datetime.now().strftime("%Y-%m-%d_%H:%M")}.npy', tokenizer,allow_pickle=True)