from model.patch_data import NDVIDataset
from model.model_2 import NDVIModel
from utils.traning import *
from utils.process_data import get_data
import pandas as pd
import datetime


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NDVIModel(sequence_length=10,img_size=(32,32))

ndvi_3d = np.load('process_data.npy')
data_cords = np.load('data_cords.npy')


train_dataset = NDVIDataset(ndvi_3d,data_cords,mode="train")
test_dataset = NDVIDataset(ndvi_3d,data_cords,mode="test")
val_dataset = NDVIDataset(ndvi_3d,data_cords,mode="val")

train_dataloader = DataLoader(train_dataset,batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_dataset,batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=8, shuffle=False)

if __name__ == "__main__":
    EPOCHS = 1
    LR = 0.001

    history,cover_model = fit(EPOCHS, LR, cover_model, train_dataloader,val_dataloader)
    
    np.save(f'history{datetime.now().strftime("%Y-%m-%d_%H:%M")}.npy', history)
    np.save(f'cover_modelhistory{datetime.now().strftime("%Y-%m-%d_%H:%M")}.npy', cover_model)