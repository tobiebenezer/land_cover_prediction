from sklearn.metrics import r2_score, mean_squared_error
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

def recursive_predictions_with_shift(model, dataloader, decoder_shape, future_steps, pred_len, model_name=None):
    model.eval()
    all_predictions = []
    all_labels = []
    all_x_dates = []
    all_y_dates = []
    all_region_ids = []

    with torch.no_grad():
        batch_count = 0
        for batch in tqdm(dataloader):
            if batch_count > 10:
                break
            X, context, [y, x_dates, y_dates, region_ids] = batch  # Assuming (X, y) format in DataLoader
            x, y = X.to(device), y.to(device)

            # Initial reshaping and encoding
            y = rearrange(y, 'b s p c h w -> b p s c h w')
           
            X, shape_x = model._encode(x)

            # Perform the initial prediction
            if model_name == 'tft':
                out = model(X, context)
            else:
                out = model(X)

            out = model._decode(out, shape_x)

            # Reshape prediction and ground truth
            b, p, s, c, h, w = y.shape
            p1 = int(p ** 0.5)
            y = rearrange(y, 'b p s c h w -> b s p c h w')
            y = rearrange(y, 'b s p c h w -> b s (p c) h w')
            y = rearrange(y, 'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

            out1 = rearrange(out, 'b p s c h w -> b s p c h w')
            out = rearrange(out1, 'b s p c h w -> b s (p c) h w')
            pred = rearrange(out, 'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)
           
            # Store initial predictions
            all_labels.append(y.cpu().numpy())
            all_x_dates.append(x_dates.cpu().numpy())
            all_y_dates.append(y_dates.cpu().numpy())
            all_region_ids.append(region_ids.cpu().numpy())

            seq_pred = [pred.cpu().numpy()[:, :pred_len]]

            # Recursive prediction loop with input shift
            for step in range(1, future_steps):
                # Shift the input:
                x = torch.cat([x[:, :-pred_len], out1[:, :pred_len]], dim=1).to(device)

                new_input, shape_x = model._encode(x)

                if model_name == 'tft':
                    out = model(new_input, context)
                else:
                    out = model(new_input)

                out = model._decode(out, shape_x)

                # Reshape for next iteration
                out1 = rearrange(out, 'b p s c h w -> b s p c h w')
                out = rearrange(out1, 'b s p c h w -> b s (p c) h w')
                pred = rearrange(out, 'b s (p1 p2) h w -> b s (p1 h) (p2 w)', p1=p1, p2=p1)

                # Append new predictions
                seq_pred.append(pred.cpu().numpy()[:, :pred_len])
                batch_count += 1

            # Concatenate all predictions for the batch
            all_predictions.append(np.concatenate(seq_pred, axis=1))

    # Concatenate all predictions for final output
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_x_dates = np.concatenate(all_x_dates, axis=0)
    all_y_dates = np.concatenate(all_y_dates, axis=0)
    all_region_ids = np.concatenate(all_region_ids, axis=0)

    return all_predictions, all_labels, all_x_dates, all_y_dates, all_region_ids


def evaluation_metric(prediction, labels):
    predictions_flat = prediction.reshape(-1, prediction.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1])

    # Compute R² Score
    r2 = r2_score(labels_flat, predictions_flat)
    print("R² Score:", r2)

    # Compute MSE
    mse = mean_squared_error(labels_flat, predictions_flat)
    print("Mean Squared Error (MSE):", mse)

    # Compute MAPE
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)  
        mask = y_true != 0  
        return (np.fabs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100  


    mape_value = mape(labels_flat, predictions_flat)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)