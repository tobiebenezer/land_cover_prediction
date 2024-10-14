from tqdm import tqdm
from tempfile import TemporaryDirectory
import os
import torch
from datetime import datetime
import time
import numpy as np
import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Learning rate scheduler
scheduler_training = optim.lr_scheduler.ReduceLROnPlateau


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader=None, opt_func=torch.optim.SGD, scheduler=None, model_name='train', accumulation_steps=1):
    since = time.time()
    
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        
        torch.save(model.state_dict(), best_model_params_path)
        best_loss = np.inf
        history = []

        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=1e-5)
        if scheduler is None:
            scheduler = scheduler_training(optimizer, mode='min', factor=0.1, patience=7)

        for epoch in tqdm(range(epochs), desc='Epoch'):
            # Training Phase
            model.train()
            train_losses = []
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                loss = model.training_step(batch, device)
                loss.backward()

                # Gradient accumulation
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_losses.append(loss.item())

            # Validation phase
            if val_loader is not None:
                result = evaluate(model, val_loader, device)
                scheduler.step(result['val_loss'])
            else:
                result = {'val_loss': 0, 'val_accuracy': 0}
            
            result['train_loss'] = np.mean(train_losses)
            model.epoch_end(epoch, result,scheduler.get_last_lr())
            history.append(result)
            
            # Save the best model
            if result['val_loss'] < best_loss:
                best_loss = result['val_loss']
                torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Loss: {best_loss:.4f}")
        
        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        
        # Save final model
        cwd = os.getcwd()
        folder = f"Trained_Models"
        final_path = os.path.abspath(os.path.join(cwd, f"../{folder}"))
        if not os.path.isdir(final_path):
            os.mkdir(final_path)
            
        final_path = f"{final_path}/{model_name}_m_{datetime.now().strftime('%Y-%m-%d')}.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Saved model at {final_path}")
        
        return history, model

# # Learning rate scheduler
# scheduler_training = optim.lr_scheduler.ReduceLROnPlateau


# #training
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)

# def fit(epochs, lr, model, train_loader, val_loader=[], opt_func= torch.optim.SGD, scheduler=None):
#         since = time.time()
        
#         #create a temporary directory to save training checkpoints
#         with TemporaryDirectory() as tempdir:
#             best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            
#             torch.save(model.state_dict(), best_model_params_path)
#             best_loss = np.inf
#             history = []
            
#             optimizer = opt_func(model.parameters(), lr=lr ,weight_decay=1e-5)
#             if scheduler != None:
#                 scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#             else:
#                 scheduler = scheduler_training(optimizer,'min', factor=0.1)

#             for epoch in tqdm(range(epochs), desc='Epoch'):
#                 #Training Phase
#                 model.train()
#                 train_losses = []
#                 for batch in train_loader:
#                     print('ok')
#                     loss = model.training_step(batch, device)
#                     train_losses.append(loss)
                    
#                     #Backpropagation
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     print('done')
                    
                    
#                 # Validation phase   
#                 result = evaluate(model, val_loader)
#                 if scheduler:
#                     scheduler.step(result['val_loss'])
#                 result['train_loss'] = torch.stack(train_losses).mean().item()
            
#                 model.epoch_end(epoch, result)
#                 history.append(result)
                
#                 # deep copy the model
#                 if result['train_loss'] < best_loss:
#                     best_loss = result['train_loss']
#                     torch.save(model.state_dict(), best_model_params_path)
                    
#             time_elapsed = time.time() - since
#             print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s" )
#             print(f"Best val Acc: {best_loss:4f}")
            
#             #load best model weights
#             model.load_state_dict(torch.load(best_model_params_path))
            
#             #save model
#             cwd = os.getcwd()
#             folder = f"Trained_Models"
#             final_path = os.path.abspath(os.path.join(cwd,f"../{folder}"))
#             if not os.path.isdir(final_path):
#                 os.mkdir(final_path)
                
#             # final_path = f"{final_path}/m_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt"
#             final_path = f"{final_path}/m_{datetime.now().strftime('%Y-%m-%d')}.pt"
#             torch.save(model.state_dict(), final_path)
#             print(f'saved model at {final_path}')
            
#             return history, model