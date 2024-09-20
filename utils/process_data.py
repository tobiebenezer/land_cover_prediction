import pandas as pd


def convert_long_lat(geo):
    geo_dict = json.loads(geo)
    return geo_dict['coordinates'], geo_dict['coordinates'][0], geo_dict['coordinates'][1] 

def get_data(path):
    df = pd.read_csv(path)
    df['coordinates'], df['longitude'], df['latitude'] = zip(*df['.geo'].apply(convert_long_lat))

    # Rename 'first' to 'NDVI' and scale the values
    df = df.rename(columns={'first': 'NDVI'})
    df['NDVI'] = df['NDVI'] / 10000
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df