import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import MBase


class CNNtokenizer(MBase):  
    def __init__(self, input_shape=(64,64), dim=[(1,128), (128,128), (128,64), (64,64)], latent_dim=256):
        super(CNNtokenizer, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        encoder_layers = []
        for in_dim, out_dim in self.dim:
            encoder_layers.extend(self.center_in(in_dim, out_dim))

        with torch.no_grad():
            dummy_input = torch.zeros(25,1, *input_shape)
            dummy_output = nn.Sequential(*encoder_layers)(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.size(0)

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(flattened_size, self.latent_dim))

        decoder_layers = [
            nn.Linear(self.latent_dim, flattened_size),
            nn.Unflatten(1, dummy_output.shape[1:])
        ]
        for in_dim, out_dim in reversed(self.dim):
            decoder_layers.extend(self.center_inv_out(out_dim, in_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Ensure input has the correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if not present
        elif len(x.shape) != 4:
            raise ValueError(f"Expected 3D or 4D input, got {len(x.shape)}D")
        print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def center_in(self, in_dim, out_dim):
        return [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        ]

    def center_inv_out(self, in_dim, out_dim):
        return [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, output_padding=0),
        ]

    def save_weights(self, encoder_path, decoder_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_weights(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

    def load_encoder_weights(self, encoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))

    def load_decoder_weights(self, decoder_path):
        self.decoder.load_state_dict(torch.load(decoder_path))

