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

class NDVIViTFT(MBase):
    def __init__(self,
        num_heads=8, 
        hidden_size=128,
        output_size=128, 
        image_size=64,
        dropout= 0.2, 
        num_patches=25,
        patch_size=3, 
        in_channel=1,
        dim=128,
        depth=2,
        heads=8, 
        mlp_ratio=4.,
        num_layers=2,
        sequence_length=16,
        pred_size=4,
        past_size=10):
        super().__init__()

        self.pred_size = pred_size
        self.sequence_length = sequence_length
        self.past_size = past_size

        self.encoder = NDVIViTEncoder(image_size=image_size,num_patches=num_patches, patch_size=patch_size, 
                        in_channel=in_channel, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)

        input_size = dim * num_patches  
        self.tft = TemporalFusionTransformer(input_size=3900, hidden_size=hidden_size, 
                        output_size=output_size, num_heads=num_heads, 
                        dropout=dropout, num_layers=num_layers, past_size=past_size,
                        patch_size=num_patches, sequence_length=sequence_length,pred_size=pred_size)

        self.decoder = NDVIViTDecoder(input_dim=output_size, output_channels=1, output_size=image_size, num_patches=num_patches)

        self.encoder2 = Sen12MSViTEncoder(image_size=image_size, num_patches=num_patches, in_channels=in_channel, dim=dim, depth=2, heads=4, mlp_ratio=2.)



    def forward(self,x, context):
       
        # encoded_output = self.encoder(x)
        encoded_output2 = self.encoder2(x)
        temporal_output, attension = self.tft(encoded_output2, context)
        output = self.decoder(temporal_output)
        
        return output, attension

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def training_step(self,batch):
        X, context, (y ,_)= batch
        out,_ = self(X.to(device), context.to(device))
        
        loss = F.mse_loss(out,y.to(device)) # calculating loss
        
        return loss
    
    def validation_step(self, batch):
        X, context, ( y,_) = batch
        out,_ = self(X.to(device), context.to(device))
        loss = F.mse_loss(out,y.to(device))
        acc = accuracy(out, y.to(device))
        return {'val_loss':loss.detach(), 'val_accuracy':acc}

