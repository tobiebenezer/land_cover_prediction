import os
import pandas as pd
from scipy import interpolate
import numpy as np
from tqdm import tqdm

class Patch_data:
    def convert_long_lat(self,geo):
        geo_dict = json.loads(geo)
        return geo_dict['coordinates'], geo_dict['coordinates'][0], geo_dict['coordinates'][1] 

    def process_df(self,data):
        data['coordinates'], data['longitude'], data['latitude'] = zip(*data['.geo'].apply(convert_long_lat))
        data.drop(columns=["system:index",".geo","coordinates"],inplace=True)
        return data

    def get_files(self,dir_path):
        files = os.listdir(dir_path)
        for folder in files:
            path = os.path.join(dir_path,folder)
            if os.path.isdir(os.path.join(dir_path,folder)):
                folder_files  = os.listdir(path)
                for file in folder_files:
                    yield os.path.join(path,file)
                    
            else:
                continue
    
    def interpolate_timeseries(self,values, kind='cubic'):
        """
        Interpolate missing values in a time series using polynomial interpolation.
        
        :param values: 2D array of NDVI values (timestamps x grid points)
        :param kind: Type of interpolation. 'linear', 'quadratic', or 'cubic'
        :return: 2D array of interpolated values
        """
        timestamps = np.arange(values.shape[0])
        interpolated = np.zeros_like(values)
        
        for i in range(values.shape[1]):
            column = values[:, i]
            mask = ~np.isnan(column)

            if np.sum(mask) > 0: 
                # Perform linear interpolation
                valid_timestamps = timestamps[mask]
                valid_values = column[mask]
                f = interpolate.interp1d(valid_timestamps, valid_values, 
                                        kind='linear', bounds_error=False, 
                                        fill_value=(valid_values[0], valid_values[-1]))
                interpolated[:, i] = f(timestamps)

                # Fill any remaining NaNs at the start with the first valid value
                first_valid_index = np.argmax(mask)
                interpolated[:first_valid_index, i] = interpolated[first_valid_index, i]

                last_valid_index = len(mask) - 1 - np.argmax(mask[::-1])
                interpolated[last_valid_index+1:, i] = interpolated[last_valid_index, i]
            else:
                interpolated[:, i] = 0.0  
        
        return interpolated

    def process_grid_chunk(self,df_chunk):
        df_chunk = df_chunk.sort_values('timestamp')
        # Extract unique timestamps and create a mapping
        timestamps = df_chunk['timestamp'].unique()
        
        # Get all unique grid points in this chunk
        all_grid_points = df_chunk[df_chunk['timestamp'] == timestamps[0]][['latitude', 'longitude']]
        num_grid_points = 1024
        
        data = []
        for timestamp in timestamps:
            time_df = df_chunk[df_chunk['timestamp'] == timestamp]
            
            # Create an array of NaNs for this timestamp
            time_NDVI = np.full(num_grid_points, np.nan)
            
            # Fill in the available NDVI values
            for grid_idx, (idx, row) in enumerate(time_df.iterrows()):
                
                time_NDVI[grid_idx] = row['NDVI']
            
            data.append(time_NDVI)
        
        ndvi_array = np.array(data)
        
        # Interpolate missing values
        ndvi_array = self.interpolate_timeseries(ndvi_array)
        
        return ndvi_array, all_grid_points

    def process_dataframe(self, df):
        grid_ids = df['grid_id'].unique()
        ndvi_arrays = []
        grid_info = []

        for grid_id in tqdm(grid_ids, desc="Grid"):
            df_chunk = df[df['grid_id'] == grid_id]
            ndvi_array, locations = self.process_grid_chunk(df_chunk)
            ndvi_arrays.append(ndvi_array)
            
            for _, location in locations.iterrows():
                grid_info.append({
                    'grid_id': grid_id,
                    'latitude': location['latitude'],
                    'longitude': location['longitude']
                })

        # Stack all NDVI arrays
        stacked_ndvi = np.stack(ndvi_arrays, axis=1)
        
        # Create a DataFrame for grid information
        grid_info_df = pd.DataFrame(grid_info)
        
        return stacked_ndvi, grid_info_df

    def get_lat_long(self, grid_index):
        grid_id = grid_info['grid_id'].unique()[grid_index]
        return grid_info[grid_info['grid_id'] == grid_id][['latitude', 'longitude']]

    def process_cords(self, ndvi_3d):
        fill = [13.1339, 27.8493]
        data_cords = []
        grid_shape = (1024,2)
        for grid_index in range(ndvi_3d.shape[1]):
            grid_cord = self.get_lat_long(grid_index).values
            
            # Create an array
            cords = np.full(grid_shape, [13, 27])
            for grid_idx, cord in enumerate(grid_cord):
                cords[grid_idx] = cord
                
            data_cords.append(cords)
    
    data_cords = np.stack(data_cords, axis=0)
    np.save('data_cords.npy', data_cords)



if __name__ == "__main__":
    
    if os.path.exists("process_data.csv"):
        df = pd.read_csv("process_data.csv")
        df = df.rename(columns = {"image_date":"timestamp"})
        ndvi_3d, grid_info = Patch_data.process_dataframe(df)    
        np.save('process_data.npy', ndvi_3d)  
        grid_info.to_csv('grid_info.csv')

    else:
        print("process_data.csv not found")
    
        