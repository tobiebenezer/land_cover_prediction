import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return torch.mean(torch.abs((labels - outputs))) * 100
