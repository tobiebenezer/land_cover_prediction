import torch
from tqdm import tqdm
from einops import rearrange
import numpy as np


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#evaluation
def compute_predictions(model, dataloader):
    model.eval()
    prediction = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            data, x = batch
            data = rearrange(data, 'b p c p1 p2 -> (b p) c p1 p2')
            pred = model(data.to(device))
            num_patchs =  x.shape[-1] // pred.shape[-1]
            pred = rearrange(pred, '(b p) c p1 p2 -> b p c p1 p2', b=x.shape[0])
            pred = rearrange(pred, 'b (h w) c p1 p2 -> (b c) (h  p1) (w  p2)', h=num_patchs, w=num_patchs )
            prediction.append(pred.cpu().data.numpy())

            labels.append(x.cpu().data.numpy())

    predictions = np.concatenate(prediction, axis=0)
    labels = np.concatenate(labels, axis=0)

    return predictions, labels