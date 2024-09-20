import sys
import os

try:
    args = sys.argv
except:
    e = sys.exc_info()[0]
    print(e)


if __name__ == "__main__":
    os.system('gdown "https://drive.google.com/uc?id=1HBAAX9gX3uWyJjkxdfjLguphdFb5cEIF" -O ndvi_sequence_samples.csv' )
    os.system('gdown "https://drive.google.com/uc?id=1HBAAX9gX3uWyJjkxdfjLguphdFb5cEIF" -O ndvi_sequence_samples.csv' )