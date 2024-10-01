import torch
from torch import nn, einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from model.TemporalTransformer.NDVIViTEncoder import *
from model.TemporalTransformer.NDVIViTDecoder import *
from model.TemporalTransformer.tft import *
from model.base import MBase, accuracy

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NDVIViTFT_tokenizer(MBase):
    def __init__(self,
        num_heads=8, 
        hidden_size=128,
        output_size=128, 
        image_size=64,
        dropout= 0.2, 
        patch_size=3, 
        in_channel=1,
        dim=128,
        depth=2,
        heads=8, 
        mlp_ratio=4.,
        num_layers=2):
        super().__init__()


        self.encoder = NDVIViTEncoder(image_size=image_size, 
                        in_channel=in_channel, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)

        self.decoder = NDVIViTDecoder(input_dim=output_size, output_channels=1, output_size=image_size)

        self.encoder2 = Sen12MSViTEncoder(image_size=image_size, in_channels=in_channel, dim=dim, depth=2, heads=4, mlp_ratio=2.)



    def forward(self,x):
       
        encoded_output = self.encoder(x)
        output = self.decoder(temporal_output)
        
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
        
        loss = F.mse_loss(out,y.to(device)) # calculating loss
        
        return loss
    
    def validation_step(self, batch):
        X, ( y,_) = batch
        out= self(X.to(device))
        loss = F.mse_loss(out,y.to(device))
        acc = accuracy(out, y.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}

