from data.feature_extraction.NDVIViTDataloader import NDIVIViTDataloader
from model.TemporalTransformer.NDVIViTEncoder import NDVIViTEncoder
from model.TemporalTransformer.NDVIViTFT_model import NDVIViTFT
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


image_size=64
num_patches=25
patch_size=3
in_channel=1
dim=128
depth=2
heads=8
mlp_ratio=4.
num_heads=8, 
hidden_size=128,
output_size=128, 
image_size=64,
dropout=0.2, 
num_patches=25,
patch_size=3, 
in_channel=1,




ndvi_3d = np.load('64x64_patches.npy')
context = np.load('context.npy')

class Scaler():
    def transform(self,x):
        return x/10000
    
    def fit_transform(self,x):
        return x/10000

    def inverse_transform(self,x):
        return x*10000

def custom_collate(batch):
    # Filter out samples with zero-sized y tensors
    batch = [item for item in batch if item[2][0].size(0) > 0]
    
    if len(batch) == 0:
        return None  
    
    x = torch.stack([item[0] for item in batch])
    context = torch.stack([item[1] for item in batch])
    y = torch.stack([item[2][0] for item in batch])
    
    return x, context, (y, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--EPOCHS', type=int, help='number of epochs')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, help='batch size')
    parser.add_argument('--SEQ_LEN', type=int, help='sequence length', default=16)
    parser.add_argument('--PRED_LEN', type=int, help='prediction length', default=4)
    parser.add_argument('--PAST_LEN', type=int, help='past length', default=10)
    parser.add_argument('--NUM_WORKERS', type=int, help='number of workers',default=1)
    args = parser.parse_args()


    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001
    BATCH_SIZE = args.BATCH_SIZE if args.BATCH_SIZE else 2
    SEQ_LEN = args.SEQ_LEN if args.SEQ_LEN else 16
    PRED_LEN = args.PRED_LEN if args.PRED_LEN else 4
    PAST_LEN = args.PAST_LEN if args.PAST_LEN else 10
    NUM_WORKERS = args.NUM_WORKERS if args.NUM_WORKERS else 1

    if not os.path.exists('scaler.pkl'):
        train_dataset = NDIVIViTDataloader(ndvi_3d,context,sequence_length=SEQ_LEN,pred_size=PRED_LEN,mode="train")
        train_dataset.save_scaler('scaler.pkl')
        scaler = train_dataset.scaler
        
    else:
        # scaler = NDIVIViTDataloader.load_scaler('scaler.pkl')
        scaler = Scaler()
        train_dataset = NDIVIViTDataloader(ndvi_3d,context,sequence_length=SEQ_LEN,pred_size=PRED_LEN,mode="train",scaler=scaler)

    test_dataset = NDIVIViTDataloader(ndvi_3d,context,sequence_length=SEQ_LEN,pred_size=PRED_LEN,mode="test",scaler=scaler)
    val_dataset = NDIVIViTDataloader(ndvi_3d,context,sequence_length=SEQ_LEN,pred_size=PRED_LEN,mode="val",scaler=scaler)


    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS,collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS,collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS,collate_fn=custom_collate)

    modelencoder = NDVIViTFT(pred_size=PRED_LEN,sequence_length=SEQ_LEN,dropout=0.3)
    modelencoder.to(device)

    history,modelencoder = fit(EPOCHS, LR, modelencoder, train_dataloader,val_dataloader)
    
    torch.save(modelencoder.state_dict(), f'cnn_modelencoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    modelencoder.save_weights(f'encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'modelencoderhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)