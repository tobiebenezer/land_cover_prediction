from re import L
from matplotlib.axis import YAxis
from model.resdecoder import ResNet18Decoder
from model.resencoder import ResNet18Encoder
from einops import rearrange
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from model.TemporalTransformer.tft import *
from model.base import MBase, accuracy
import torch.nn as nn
import torch.optim as optim
from metrics.loss_func import mse_loss, combined_loss
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Combine_model(MBase):
    def __init__(self, model,encoder,input_size,model_param, pred_size,sequence_length, in_channels=1, out_channels=1):
        super(Combine_model, self).__init__()

        self.pred_size = pred_size
        self.ae_model = encoder['model']()
        self.ae_model.load_state_dict(torch.load(encoder['parameter_path']))
        self.ae_model.to(encoder['device'])

        model_param[-1] = input_size
        self.model = model['model'](*model_param, input_size = input_size,pred_size=pred_size, sequence_length=sequence_length).to(device)  
        self.model.to(model['device'])

    def forward(self, x):
        b,s,p,c,h,w = x.shape
        x = rearrange(x,'b s p c h w -> b p s c h w')
        x = rearrange(x,'b p s c h w -> (b p s) c h w')
        with torch.no_grad():
            x4 = self.ae_model.encoder(x)

        x4 = rearrange(x4, '(b p s) c h w->(b p) s (c h w)', s=s, b=b, p=p)
        latent_space_pred = self.model(x4)
        latent_space_pred = latent_space_pred[:,:self.pred_size,:]
        
        # latent_space_pred = latent_space_pred.reshape([*x_dim])

        # with torch.no_grad():
        #     output = self.ae_model.decoder(latent_space_pred)

        # output = rearrange(output,'(b s) c h w  -> b c s h w', b=b )
        # output = output[:,:, output.shape[2] - self.pred_size:,:,:]
        return latent_space_pred

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def training_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
        X, y = X.to(device), y.to(device)
        
        with torch.no_grad():
            b,s,p,c,h,w = y.shape
            y = rearrange(y,'b s p c h w -> b p s c h w')
            y = rearrange(y,'b p s c h w -> (b p s) c h w')
            y = self.ae_model.encoder(y)
            y = rearrange(y, '(b p s) c h w->(b p) s (c h w)', s=s, b=b, p=p)

        # Forward pass
        out = self(X)        
        # Sum of Squared Errors Loss (SSE)
        loss = mse_loss(out, y)
        
        return loss

    def validation_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            b,s,p,c,h,w = y.shape
            y = rearrange(y,'b s p c h w -> b p s c h w')
            y = rearrange(y,'b p s c h w -> (b p s) c h w')
            y = self.ae_model.encoder(y)
            y = rearrange(y, '(b p s) c h w->(b p) s (c h w)', s=s, b=b, p=p)
        
        # Forward pass
        out = self(X)
        # Loss and accuracy
        loss = mse_loss(out, y)
        acc = self.accuracy(out, y)

        return {'val_loss': loss.detach(), 'val_accuracy': acc}

    def validation_epoch_end(self, outputs):
        # Collecting batch losses and accuracies
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [x['val_accuracy'] for x in outputs]

        # Averaging loss and accuracy over the epoch
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_accuracy).mean()

        return {'val_loss': epoch_loss.item(), 'val_accuracy': epoch_acc.item()}
    
    def accuracy(self, outputs, labels):
        outputs_flat = outputs.reshape(-1)
        labels_flat = labels.reshape(-1)
        return 1 - MeanAbsoluteError().to(device)(outputs, labels)