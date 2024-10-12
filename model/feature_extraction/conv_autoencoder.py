"""
Archtecture as describe in 
L. Bergamasco, S. Saha, F. Bovolo and L. Bruzzone, "Unsupervised Change Detection 
Using Convolutional-Autoencoder Multiresolution Features," in IEEE Transactions on Geoscience and Remote Sensing,
 vol. 60, pp. 1-19, 2022, Art no. 4408119, doi: 10.1109/TGRS.2022.3140404.
keywords: {Feature extraction;Training;Data models;Decoding;Task analysis;Remote
 sensing;Semantics;Convolutional autoencoder (CAE);deep learning (DL);multitemporal analysis;remote sensing (RS);unsupervised change detection (CD);unsupervised learning},
"""
from model.base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from metrics.loss_func import mse_loss, combined_loss

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.enc2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) 
        self.bn2 = nn.BatchNorm2d(64)  
        self.enc3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.enc1(x)))
        x = self.leaky_relu(self.bn2(self.enc2(x)))
        x = self.leaky_relu(self.bn3(self.enc3(x)))
        return x


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)  # Deconv2D layer
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)  # Deconv2D layer
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)  # Deconv2D layer (final layer)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.dec1(x)))  
        x = self.leaky_relu(self.bn2(self.dec2(x)))  
        x = self.dec3(x)  
        return x

class CAE(MBase):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    
    def accuracy(outputs, labels, device):
        outputs_flat = outputs.reshape(-1)
        labels_flat = labels.reshape(-1)
        return 1 - MeanAbsoluteError().to(device)(outputs_flat, labels_flat)



if __name__ == "__main__":
    model = CAE()
    print(model)

    x = torch.randn(8, 1, 64, 64)
    output(model(x))
    print(output.shape,'output shape')
