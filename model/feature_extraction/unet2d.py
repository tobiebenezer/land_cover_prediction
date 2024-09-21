import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import MBase
import os
import tempfile
import shutil
from datetime import datetime



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

    # def save_weights(self, encoder_path, decoder_path):
    #     torch.save(self.encoder.state_dict(), encoder_path)
    #     torch.save(self.decoder.state_dict(), decoder_path)
    def save_weights(self, encoder_path, decoder_path):
        def safe_save(state_dict, path):
            try:
                # First, try to save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    torch.save(state_dict, tmp_file.name)
                
                # If successful, move the temporary file to the desired location
                shutil.move(tmp_file.name, path)
                print(f"Successfully saved weights to {path}")
            except Exception as e:
                print(f"Error saving to {path}: {str(e)}")
                
                # Fallback: Try saving to the current directory with a timestamp
                fallback_path = f"weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                try:
                    torch.save(state_dict, fallback_path)
                    print(f"Saved weights to fallback location: {fallback_path}")
                except Exception as e2:
                    print(f"Failed to save weights to fallback location: {str(e2)}")
                    
                    # Last resort: Try saving to system's temporary directory
                    temp_dir = tempfile.gettempdir()
                    last_resort_path = os.path.join(temp_dir, f"weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                    try:
                        torch.save(state_dict, last_resort_path)
                        print(f"Saved weights to temporary directory: {last_resort_path}")
                    except Exception as e3:
                        print(f"Failed to save weights to temporary directory: {str(e3)}")
                        print("Unable to save weights. Please check your disk space and permissions.")

    safe_save(self.encoder.state_dict(), encoder_path)
    safe_save(self.decoder.state_dict(), decoder_path)

    def load_weights(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

    def load_encoder_weights(self, encoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))

    def load_decoder_weights(self, decoder_path):
        self.decoder.load_state_dict(torch.load(decoder_path))

