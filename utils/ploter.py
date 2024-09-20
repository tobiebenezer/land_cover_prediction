from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def plot_map(long,lat,nvdi,region):
    
    # Plot the first time step
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = LinearSegmentedColormap.from_list('ndvi', ['brown', 'yellow', 'green'])
   
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