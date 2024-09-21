def get_patch_indices(s, b, patch_size=2):
    """
    Extract the 2x2 patch index from the unsord grid_id 

    :param s: unsorded grid_id
    :param b: sorded gridO_id
    :param patch_size: int size of patch

    :return: list of patch indices
    
    """
    b_grid = b.reshape(10, 10)
    
    s_index_map = {val: idx for idx, val in enumerate(s)}
    
    result = []
    
    for i in range(0, 10 - patch_size + 1,patch_size):
        for j in range(0, 10 - patch_size + 1,patch_size):
            patch = b_grid[i:i+patch_size, j:j+patch_size].flatten()
            
            # Find the corresponding indices in s
            patch_indices = [s_index_map[val] for val in patch]
            result.append(patch_indices)
    
    return result

def extract_2x2_patches(ndvi_3d, patch_indices):
    """
    Extract 2x2 patches from a 3D array based on given indices.
    
    :param ndvi_3d: NDVI 
    :param patch_indices: List of interger contain the in of the rows to extract
    :return: Numpy array containing the extracted 2x2 patches
    """
    timestamp_data = []
    for ndvi in ndvi_3d:
        patch_data = []
        ndvi = ndvi.reshape(ndvi.shape[0],32,32)
        for patch in patch_indices:
                        
            x1 = np.hstack((ndvi[patch[0]],ndvi[patch[1]]))
            x2 = np.hstack((ndvi[patch[2]],ndvi[patch[3]]))
            x = np.vstack((x1,x2))
            patch_data.append(x)
            
        timestamp_data.append(np.stack(patch_data))
        
    return np.stack(timestamp_data)

if __name__ == "__main__":
    ndvi_3d = np.load('process_data.npy')
    grid_info = pd.read_csv('grid_info.csv')
    s = grid_info['grid_id'].unique()
    b = np.sort(s)
    patch_indices = get_patch_indices(s, b)
    patches = extract_2x2_patches(ndvi_3d, patch_indices)
    np.save('64x64_patches.npy', patches)
    print(patches.shape)