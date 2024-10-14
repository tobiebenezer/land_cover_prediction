from pytorch_msssim import ssim
from torchmetrics.functional import mean_absolute_error as MeanAbsoluteError
from einops import rearrange
import torch.nn.functional as F

def mae_loss(recon_x, x):
    return F.l1_loss(recon_x, x)

def mse_loss(recon_x, x):
    return F.mse_loss(recon_x, x)

def ssim_loss(recon_x, x):
    recon_x = recon_x.unsqueeze(1)
    x = x.unsqueeze(1)
    return 1 - ssim(recon_x, x, data_range=1, size_average=True)

def combined_loss(recon_x, x, alpha=0.5):
    mse = mse_loss(recon_x, x)
    ssim = ssim_loss(recon_x, x)
    return alpha * mse + (1 - alpha) * ssim

def huber_loss(recon_x, x, delta=1.0):
    error = recon_x - x
    is_small_error = torch.abs(error) <= delta
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

def sse_loss(recon_x, x):
    return (recon_x - x).pow(2).sum()

def accuracy(outputs, labels, device):
    outputs_flat = outputs.reshape(-1)
    labels_flat = labels.reshape(-1)
    return 1 - MeanAbsoluteError().to(device)(outputs_flat, labels_flat)