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
    def __init__(self, model,encoder,input_size,model_param, pred_size,sequence_length,hidden_dim=512, in_channels=1, out_channels=1):
        super(Combine_model, self).__init__()

        self.pred_size = pred_size
        self.ae_model = encoder['model'](hidden_dim)
        self.ae_model.load_state_dict(torch.load(encoder['parameter_path']))
        self.ae_model.to(encoder['device'])

        model_param[-1] = input_size
        self.model = model['model'](*model_param, input_size = input_size,pred_size=pred_size, sequence_length=sequence_length).to(device)  
        self.model.to(model['device'])
    
    def forward(self, x):
        latent_space_pred = self.model(x)
        latent_space_pred = latent_space_pred[:,:self.pred_size,:]
                
        return latent_space_pred 

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def _encode(self,x):
        b,s,p,c,h,w = x.shape
        x = rearrange(x,'b s p c h w -> b p s c h w')
        x = rearrange(x,'b p s c h w -> (b p s) c h w')
       
        x= self.ae_model.encoder(x)
        _,c2,p1,p2 = x.shape
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.ae_model.leaky_relu(self.ae_model.fc1(x))
        
        x = rearrange(x, '(b p s) h ->(b p) s h', s=s, b=b, p=p)
        return x,[b, p, c2, p1, p2 ]

    def _decode(self,x, shape_x):
        [b, p, c2, p1, p2] = shape_x
        
        latent_space_pred = rearrange(x, '(b p) s h ->(b p s) h', s=self.pred_size, b=b, p=p)

        latent_space_pred = self.ae_model.leaky_relu(self.ae_model.fc2(latent_space_pred))
        b2,_ = latent_space_pred.shape
        latent_space_pred = rearrange(latent_space_pred, 'b (c h w) -> b c h w', b=b2, c=c2, h=p1, w=p2)
        output = self.ae_model.decoder(latent_space_pred)
        
        out = rearrange(output,'(b p s) c h w -> b p s c h w',  s=self.pred_size, b=b, p=p)
        return out

    def training_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
        x, y = X.to(device), y.to(device)

        y = rearrange(y,'b s p c h w -> b p s c h w')
        X, shape_x = self._encode(x)
        # Forward pass
        out = self(X)  
        out = self._decode(out,shape_x)

        b ,p ,s ,c ,h, w = y.shape
        p1 = int(p**0.5) 
        y = rearrange(y,'b p s c h w -> b s p c h w')
        y = rearrange(y,'b s p c h w -> b s (p c) h w')
        y = rearrange(y,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        out = rearrange(out,'b p s c h w -> b s p c h w')
        out = rearrange(out,'b s p c h w -> b s (p c) h w')
        out = rearrange(out,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        # loss = mse_loss(out, y)
        loss = combined_loss(out, y)
        
        return loss

    def validation_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
        x, y = X.to(device), y.to(device)

        y = rearrange(y,'b s p c h w -> b p s c h w')
        X, shape_x = self._encode(x)
        # Forward pass
        out = self(X)  
        out = self._decode(out,shape_x)        
        
        b ,p ,s ,c ,h, w = y.shape
        p1 = int(p**0.5) 
        y = rearrange(y,'b p s c h w -> b s p c h w')
        y = rearrange(y,'b s p c h w -> b s (p c) h w')
        y = rearrange(y,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        out = rearrange(out,'b p s c h w -> b s p c h w')
        out = rearrange(out,'b s p c h w -> b s (p c) h w')
        out = rearrange(out,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        # loss = mse_loss(out, y)
        loss = combined_loss(out, y)
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


class Combine_transformer_model(MBase):
    def __init__(self, model,encoder,input_size,model_param, pred_size,sequence_length,model_name,hidden_dim=512, in_channels=1, out_channels=1):
        super(Combine_transformer_model, self).__init__()

        self.pred_size = pred_size
        self.model_name = model_name

        self.ae_model = encoder['model'](hidden_dim)
        self.ae_model.load_state_dict(torch.load(encoder['parameter_path']))
        self.ae_model.to(encoder['device'])

        model_param[-1] = input_size
        if self.model_name == 'tft':
            num_heads = 8
            dropout = 0.2
            num_layers = 2
            model[0] = input_size
            self.model = model['model'](*model_param, num_heads=num_heads, dropout=dropout, num_layers=num_layers)
        else:
            self.model = model['model'](*model_param, input_size = input_size,pred_size=pred_size, sequence_length=sequence_length).to(device)  
        self.model.to(model['device'])

    def forward(self, x):
        if self.model_name == 'tft':
            latent_space_pred, attns = self.model(latent_space)
        else:
            latent_space_pred  = self.model(latent_space)

        latent_space_pred = latent_space_pred[:,:self.pred_size,:]
        
        return latent_space_pred

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_state_dict(torch.load(encoder_weights))
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def save_weights(self, encoder_weights, decoder_weights):
        torch.save(self.encoder.state_dict(), encoder_weights)
        torch.save(self.decoder.state_dict(), decoder_weights)

    def _encode(self,x):
        b,s,p,c,h,w = x.shape
        x = rearrange(x,'b s p c h w -> b p s c h w')
        x = rearrange(x,'b p s c h w -> (b p s) c h w')
       
        x= self.ae_model.encoder(x)
        _,c2,p1,p2 = x.shape
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.ae_model.leaky_relu(self.ae_model.fc1(x))
        
        x = rearrange(x, '(b p s) h ->(b p) s h', s=s, b=b, p=p)
        return x,[b, p, c2, p1, p2 ]

    def _decode(self,x, shape_x):
        [b, p, c2, p1, p2] = shape_x
        
        latent_space_pred = rearrange(x, '(b p) s h ->(b p s) h', s=self.pred_size, b=b, p=p)

        latent_space_pred = self.ae_model.leaky_relu(self.ae_model.fc2(latent_space_pred))
        b2,_ = latent_space_pred.shape
        latent_space_pred = rearrange(latent_space_pred, 'b (c h w) -> b c h w', b=b2, c=c2, h=p1, w=p2)
        output = self.ae_model.decoder(latent_space_pred)
        
        out = rearrange(output,'(b p s) c h w -> b p s c h w',  s=self.pred_size, b=b, p=p)
        return out

    def training_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
        x, y = X.to(device), y.to(device)

        y = rearrange(y,'b s p c h w -> b p s c h w')
        X, shape_x = self._encode(x)
        # Forward pass
        if self.model_name == 'tft':
            out = self.model(X, context)
        else:
            out = self(X) 

        out = self._decode(out,shape_x)        

        b ,p ,s ,c ,h, w = y.shape
        p1 = int(p**0.5) 
        y = rearrange(y,'b p s c h w -> b s p c h w')
        y = rearrange(y,'b s p c h w -> b s (p c) h w')
        y = rearrange(y,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        out = rearrange(out,'b p s c h w -> b s p c h w')
        out = rearrange(out,'b s p c h w -> b s (p c) h w')
        out = rearrange(out,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        # loss = mse_loss(out, y)
        loss = combined_loss(out, y)
        
        return loss

    def validation_step(self, batch, device):
        X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
        x, y = X.to(device), y.to(device)

        y = rearrange(y,'b s p c h w -> b p s c h w')
        X, shape_x = self._encode(x)
        # Forward pass
        if self.model_name == 'tft':
            out = self.model(X, context)
        else:
            out = self(X)

        out = self._decode(out,shape_x)        

        b ,p ,s ,c ,h, w = y.shape
        p1 = int(p**0.5) 
        y = rearrange(y,'b p s c h w -> b s p c h w')
        y = rearrange(y,'b s p c h w -> b s (p c) h w')
        y = rearrange(y,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        out = rearrange(out,'b p s c h w -> b s p c h w')
        out = rearrange(out,'b s p c h w -> b s (p c) h w')
        out = rearrange(out,'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

        # loss = mse_loss(out, y)
        loss = combined_loss(out, y)
        acc = self.accuracy(out, y)

        return {'val_loss': loss.detach(), 'val_accuracy': acc}

    # def training_step(self, batch, device):
    #     X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
    #     X, y = X.to(device), y.to(device)
    #     y = rearrange(y,'b s p c h w -> b p s c h w')

        
    #     # with torch.no_grad():
    #     #     b,s,p,c,h,w = y.shape
    #     #     y = rearrange(y,'b s p c h w -> b p s c h w')
    #         # y = rearrange(y,'b p s c h w -> (b p s) c h w')
    #         # y = self.ae_model.encoder(y)
    #         # y = rearrange(y, '(b p s) c h w->(b p) s (c h w)', s=s, b=b, p=p)

    #     # Forward pass
    #     out = self(X)        
    #     # Sum of Squared Errors Loss (SSE)
        loss = mse_loss(out, y)
        loss = combined_loss(out, y)
        
    #     return loss

    # def validation_step(self, batch, device):
    #     X, context, [y, x_dates, y_dates, region_ids] = batch
    #     X, y = X.to(device), y.to(device)
    #     y = rearrange(y,'b s p c h w -> b p s c h w')

    #     # with torch.no_grad():
    #     #     b,s,p,c,h,w = y.shape
    #     #     y = rearrange(y,'b s p c h w -> b p s c h w')
    #         # y = rearrange(y,'b p s c h w -> (b p s) c h w')
    #         # y = self.ae_model.encoder(y)
    #         # y = rearrange(y, '(b p s) c h w->(b p) s (c h w)', s=s, b=b, p=p)
        
    #     # Forward pass
    #     if self.model_name == 'tft':
    #         out = self.model(X, context)
    #     else:
    #         out = self(X)
    #     # Loss and accuracy
        loss = mse_loss(out, y)
        loss = combined_loss(out, y)
    #     acc = self.accuracy(out, y)

    #     return {'val_loss': loss.detach(), 'val_accuracy': acc}

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
