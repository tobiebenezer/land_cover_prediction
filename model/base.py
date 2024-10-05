import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from einops import rearrange


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class MBase(nn.Module):
    def training_step(self,batch):
        X, (y ,_)= batch
        out = self(X.to(device))
        loss = F.mse_loss(out,y.to(device)) # calculating loss
      
        return loss
    
    def validation_step(self, batch):
        X,( y,_) = batch
        out = self(X.to(device))
        loss = F.mse_loss(out,y.to(device))
        acc = accuracy(out, y.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [x['val_accuracy'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_accuracy).mean()
        return {'val_loss': epoch_loss.item(),'val_accuracy':epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_accuracy']))
        

def accuracy(outputs, labels):
    # return torch.mean(torch.abs((labels - outputs))) * 100
    val_mae = MeanAbsoluteError().to(device)
    # val_mse = MeanSquaredError().to(device)
    outputs_flat = outputs.reshape(-1)
    labels_flat = labels.reshape(-1)

    return val_mae(outputs_flat, labels_flat)

def rmse(outputs, labels):
    mse = accuracy(outputs, labels)
    return torch.sqrt(mse)

def mean_absolute_error(outputs, labels):
    outputs_flat = outputs.reshape(outputs.size(0), -1)
    labels_flat = labels.reshape(labels.size(0), -1)
    return torch.mean(torch.abs(outputs_flat - labels_flat))