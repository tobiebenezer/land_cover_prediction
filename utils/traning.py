from tqdm import tqdm
from tempfile import TemporaryDirectory
import os
import torch
from datetime import datetime
import time
import numpy as np


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#training
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader=[], opt_func= torch.optim.SGD, scheduler=None):
        since = time.time()
        
        #create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            
            torch.save(model.state_dict(), best_model_params_path)
            best_loss = np.inf
            history = []
            
            optimizer = opt_func(model.parameters(), lr)
#             if scheduler:
#                 scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                
            for epoch in tqdm(range(epochs), desc='Epoch'):
                #Training Phase
                model.train()
                train_losses = []
                for batch in train_loader:
                    loss = model.training_step(batch)
                    train_losses.append(loss)
                    break
                    #Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if scheduler:
                    scheduler.step()
                    
                # Validation phase   
                result = evaluate(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
            
                model.epoch_end(epoch, result)
                history.append(result)
                
                # deep copy the model
                if result['train_loss'] < best_loss:
                    best_loss = result['train_loss']
                    torch.save(model.state_dict(), best_model_params_path)
                    
            time_elapsed = time.time() - since
            print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s" )
            print(f"Best val Acc: {best_loss:4f}")
            
            #load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
            
            #save model
            cwd = os.getcwd()
            folder = f"Trained_Models"
            final_path = os.path.abspath(os.path.join(cwd,f"../{folder}"))
            if not os.path.isdir(final_path):
                os.mkdir(final_path)
                
            final_path = f"{final_path}/m_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt"
            torch.save(model.state_dict(), final_path)
            
            return history, model