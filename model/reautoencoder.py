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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class ResNet18Autoencoder(MBase):
    def __init__(self, in_channels=1, out_channels=3):
        super(ResNet18Autoencoder, self).__init__()
        self.encoder = ResNet18Encoder(in_channels)
        self.decoder = ResNet18Decoder(out_channels)

    def forward(self, x):
        x4,_ = self.encoder(x)
        x4 = x4.reshape(self.encoder.output_dim)
        output = self.decoder(x4)
        return output

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def training_step(self,batch):
        X, (y ,_)= batch
        out = self(X.to(device))
        out = rearrange(out, 'b c h w -> (b c) h w')
        loss = F.mse_loss(out,X.to(device)) # calculating loss
        
        return loss
    
    def validation_step(self, batch):
        X, ( y,_) = batch
        out= self(X.to(device))
        out = rearrange(out, 'b c h w -> (b c) h w')
        loss = F.mse_loss(out,X.to(device))
        acc = accuracy(out, X.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}



