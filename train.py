from model.data import NDVIDataset
from model.ndvi_model import NDVIModel
from utils.traning import *
from utils.process_data import get_data
import pandas as pd


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NDVIModel(sequence_length=10,img_size=(32,32))
df =  get_data('ndvi_sequence_samples.csv')

train_dataset = NDVIDataset(df,mode="train")
val_dataset = NDVIDataset(df,mode="val")

train_dataloader = DataLoader(train_dataset,batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_dataset,batch_size=8, shuffle=False)

if __name__ == "__main__":
    EPOCHS = 1
    LR = 0.0001

    cover_model = fit(EPOCHS, LR, model, train_dataloader,val_dataloader)