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
        X, y = batch
        out = self(X.to(device))
        loss = F.mse_loss(out,out) # calculating loss
        print(out,y)
        return loss
    
    def validation_step(self, batch):
        X, y = batch
        out = self(X.to(device))
        loss = F.mse_loss(out,y)
        acc = accuracy(out, y.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))
        

def accuracy(outputs, labels):
    return torch.mean(torch.abs((labels - outputs) / labels)) * 100
