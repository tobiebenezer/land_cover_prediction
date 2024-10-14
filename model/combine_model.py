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
    def __init__(self, model,ae_model,input_size,model_param, pred_size,sequence_length, in_channels=1, out_channels=1):
        super(Combine_model, self).__init__()

        self.pred_size = pred_size
        self.ae_model = ae_model
        model_param[-1] = input_size
        self.model = model(*model_param, input_size = input_size,pred_size=pred_size, sequence_length=sequence_length).to(device)  
          

    def forward(self, x):
        b,s,p,_,_,_ = x.shape
        x = rearrange(x,'b s p c h w -> b p s c h w')
        x = rearrange(x,'b p s c h w -> (b p s) c h w')
        with torch.no_grad():
            x4 = self.ae_modelencoder(x)

        x4 = rearrange(x4, '(b p s) c h w->(b p) s (c h w)', s=s)
        latent_space_pred = self.model(x4)
        # latent_space_pred = latent_space_pred.reshape([*x_dim])

        # with torch.no_grad():
        #     output = self.ae_modeldecoder(latent_space_pred)

        # output = rearrange(output,'(b s) c h w  -> b c s h w', b=b )
        # output = output[:,:, output.shape[2] - self.pred_size:,:,:]
        return latent_space_pred, x4

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def training_step(self,batch):
        X,context, (y ,_)= batch
       
        out = self(X.to(device))
       
        loss = F.mse_loss(out,y.to(device)) # calculating loss
        
        return loss
    
    def validation_step(self, batch):
        X, context,( y,_) = batch
        out= self(X.to(device))
        loss = F.mse_loss(out,y.to(device))
        acc = accuracy(out, y.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}
