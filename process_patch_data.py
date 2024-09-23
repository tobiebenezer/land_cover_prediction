import data.patch_data as patch_data
import pandas as pd
import os

if __name__ == "__main__":
    patch_data = patch_data.Patch_data()
    
    if not os.path.exists("NDVI_Diffusion_Data_2"):
        os.system('gdown gdown 1j9PE1WrixaLBNetyGvA6y0D_1dILsEYb --remaining-ok' )
    
    folder_path = "NDVI_Diffusion_Data_2/"

    if os.path.exists(folder_path):
        #load all the item in a folder
        files_count = 0
        df = None

        #for all item load file into csv file
        for file_path in get_files(folder_path):
            if files_count == 0:
                df = patch_data.process_df(pd.read_csv(file_path))
                
            else:
                i_df = patch_data.process_df(pd.read_csv(file_path))
                df = pd.concat([df,i_df],ignore_index=True).sort_values(by="image_date")
                if(files_count % 12) == 0:
                    print(file_path,df.count()["grid_id"])
            files_count += 1

        df.to_csv("process_data.csv", index=False)