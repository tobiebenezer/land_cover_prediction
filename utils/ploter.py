from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os


palette = ['#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
       '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
       '#012e01', '#011d01', '#011301']

cmap = cmap = LinearSegmentedColormap.from_list('custom_ndvi', palette, N=len(palette))

def plot_map(long,lat,nvdi,region):
    
    # Plot the first time step
    fig, ax = plt.subplots(figsize=(4, 4))
   
    # Plot Zambia
    # Plot NDVI points
    scatter = ax.scatter(long, lat, c=nvdi, 
                         cmap=cmap , vmin=-1, vmax=1, s=0.2)
    region.plot(ax=ax, color='none', edgecolor='black')

    plt.title(f"NDVI Map ")
    plt.colorbar(scatter, label='NDVI')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_reconstructed_img(reconstructed_image):
    # Display the reconstructed image for the first sequence
    plt.imshow(reconstructed_image, cmap=cmap)  # First image in the batch, first time step
    plt.title("Reconstructed NDVI Image")
    plt.tight_layout()
    plt.show()

def plot_comparism(pred, label):
    # Compute the reconstruction error (e.g., absolute difference)
    reconstruction_error = np.abs(pred - label)

    # Set up the subplots with 1 row and 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the predicted (reconstructed) cover
    scatter1 = ax[0].imshow(pred, cmap=cmap)
    ax[0].set_title("Reconstructed Cover")
    ax[0].axis('off')
    fig.colorbar(scatter1, ax=ax[0], fraction=0.046, pad=0.04)

    # Plot the original cover
    scatter2 = ax[1].imshow(label, cmap=cmap)
    ax[1].set_title("Original Cover")
    ax[1].axis('off')
    fig.colorbar(scatter2, ax=ax[1], fraction=0.046, pad=0.04)

    # Plot the reconstruction error
    scatter3 = ax[2].imshow(reconstruction_error, cmap='hot_r')
    ax[2].set_title("Reconstruction Error")
    ax[2].axis('off')
    fig.colorbar(scatter3, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def all_plot_comparism(pred, label,m_name, output_dir="/plots"):
    # Ensure pred and label have the same shape
    assert pred.shape == label.shape, "Predicted and label shapes must match."

    b, s, h, w = pred.shape 
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(b, 3, figsize=(12, 4 * b))

    for i in range(b):
        for t in range(s):
            # Compute the reconstruction error for the current batch and time step
            reconstruction_error = np.abs(pred[i, t] - label[i, t])

            # Plot the predicted (reconstructed) cover
            scatter1 = axes[i, 0].imshow(pred[i, t], cmap=cmap) 
            axes[i, 0].set_title(f"Reconstructed Cover (Batch {i}, Time {t})")
            axes[i, 0].axis('off')
            fig.colorbar(scatter1, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # Plot the original cover
            scatter2 = axes[i, 1].imshow(label[i, t], cmap=cmap)
            axes[i, 1].set_title(f"Original Cover (Batch {i}, Time {t})")
            axes[i, 1].axis('off')
            fig.colorbar(scatter2, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Plot the reconstruction error
            scatter3 = axes[i, 2].imshow(reconstruction_error, cmap='hot_r')
            axes[i, 2].set_title(f"Reconstruction Error (Batch {i}, Time {t})")
            axes[i, 2].axis('off')
            fig.colorbar(scatter3, ax=axes[i, 2], fraction=0.046, pad=0.04)

    # Save the figure as a PDF
    pdf_path = os.path.join(output_dir, f"{m_name}_comparison_plots.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Plots saved to {pdf_path}")

    plt.tight_layout()
    plt.show()



# def plot_comparism(pred, label):
#     fig, ax = plt.subplots(1,2,figsize=(8,4))
#     scatter1 =ax[0].imshow(pred, cmap=cmap)

#     ax[0].set_title("Reconstructed Cover")

#     scatter2 = ax[1].imshow(label, cmap=cmap)

#     ax[1].set_title("Original Cover")

#     ax[0].axis('off')
#     ax[1].axis('off')
#     plt.tight_layout()
#     plt.show()

def plot_loss(history):
    val_loss = [train['val_loss'] for train in history]
    train_loss = [train['train_loss'] for train in history]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(range(0,len(history)), val_loss, c='b', label="val loss")
    ax.plot(range(0,len(history)), train_loss, c='orange', label="train loss")

    plt.title('Training Loss')
    plt.tight_layout()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def plot_ndvi(actual, pred,period=100):
    plt.figure(figsize=(12, 6))
    plt.plot(actual[:period], label='Actual', alpha=0.7)
    plt.plot(pred[:period], label='Predicted', alpha=0.7)
    plt.title("NDVI: Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("NDVI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ndvi_time_series(train, test=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train['NDVI'],linestyle='-', color='b')
    if test is not None:
        plt.plot(test['NDVI'],linestyle='-', color='orange')
    plt.title('NDVI Time Series')
    plt.xlabel('Date')
    plt.ylabel('NDVI')
    # plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # plt.xticks(rotation=45)
    plt.legend(['Training', 'Testing'])
    plt.tight_layout()
