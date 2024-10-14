from model.resdecoder import ResNet18Decoder
from model.resencoder import ResNet18Encoder
from einops import rearrange
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from model.TemporalTransformer.tft import *
from model.base import MBase, accuracy
from metrics.loss_func import mse_loss, combined_loss
from torchmetrics import MeanAbsoluteError, MeanSquaredError
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

    def training_step(self, batch, device):
            X, y = batch  # Assuming (X, y) format in DataLoader
            X, y = X.to(device), y.to(device)

            X = rearrange(X, 'b p c p1 p2 -> (b p) c p1 p2')
            # Forward pass
            out = self(X)
            num_patchs =  y.shape[-1] // out.shape[-1]
            out = rearrange(out, '(b p) c p1 p2 -> b p c p1 p2', b=y.shape[0])
            out = rearrange(out, 'b (h w) c p1 p2 -> (b c) (h  p1) (w  p2)', h=num_patchs, w=num_patchs )

            # Sum of Squared Errors Loss (SSE)
            # loss = mse_loss(out, y)
            loss = combined_loss(out, y)
            return loss

    def validation_step(self, batch, device):
        X, y = batch
        X, y = X.to(device), y.to(device)
        X = rearrange(X, 'b p c p1 p2 -> (b p) c p1 p2')
        # Forward pass
        out = self(X)
        num_patchs =  y.shape[-1] // out.shape[-1]
        out = rearrange(out, '(b p) c p1 p2 -> b p c p1 p2', b=y.shape[0])
        out = rearrange(out, 'b (h w) c p1 p2 -> (b c) (h  p1) (w  p2)', h=num_patchs, w=num_patchs )

        # Loss and accuracy
        # loss = mse_loss(out, y)
        loss = combined_loss(out, y)
        acc = self.accuracy(out, y, device)

        return {'val_loss': loss.detach(), 'val_accuracy': acc}

    def validation_epoch_end(self, outputs):
        # Collecting batch losses and accuracies
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [x['val_accuracy'] for x in outputs]

        # Averaging loss and accuracy over the epoch
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_accuracy).mean()

        return {'val_loss': epoch_loss.item(), 'val_accuracy': epoch_acc.item()}
    
    def accuracy(self, outputs, labels, device):
        outputs_flat = outputs.reshape(-1)
        labels_flat = labels.reshape(-1)
        return 1 - MeanAbsoluteError().to(device)(outputs_flat, labels_flat)



