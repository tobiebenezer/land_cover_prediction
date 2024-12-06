from data.ndvi_dataset import NDVIDataset
import torch
from torch.utils.data import Dataset, DataLoader
from model.combine_model import  Combine_model, Combine_transformer_model
from model.lstm_model import LSTM
from model.rnn_model import SRNN
from model.gru_model import GRU
from model.TemporalTransformer.tft import TemporalFusionTransformer as TFT
from model.patch_transformer import PatchTST 
from model.feature_extraction.conv_autoencoder import CAE
from model.feature_extraction.unet2d import CNNtokenizer
from model.TemporalTransformer.tokenizer import NDVIViTFT_tokenizer
from model.reautoencoder import ResNet18Autoencoder
from model.vit_autoencoder import ViTAutoencoder
from utils.traning import *
from utils.process_data import get_data
from utils.dataloader import get_dataloaders, get_dataloaders_2
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

basemodels = {
   'CAE' : {
    'model': CAE,
    'parameter_path': '/content/Trained_Models/CAE_m_2024-10-13.pt',
    'device': device
    },
    'GRU': {
    'model': GRU,
    'parameter_path': None,
    'device': device
    },
    'LSTM': {
    'model': LSTM,
    'parameter_path': None,
    'device': device
    },
    'SRNN': {
    'model': SRNN,
    'parameter_path': None,
    'device': device
    },
    'tft' : {
    'model': TFT,
    'parameter_path': None,
    'device': device
    },
    'patchTST':{
    'model': PatchTST,
    'parameter_path': None,
    'device': device
    }
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training tokenizer \n MODEL_NAME: CNNtokenizer, custom_CNNtokenizer, NDVIViTFT_tokenizer, ResNet18Autoencoder')
    parser.add_argument('--EPOCHS', type=int, help='number of epochs')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, help='batch size')
    parser.add_argument('--IMAGES_LOG', type=str, help='path to images log',default='processed_images_log.csv')
    parser.add_argument('--DATA_DIR', type=str, help='path to data directory',default='extracted_data')
    parser.add_argument('--NUM_WORKERS', type=int, help='number of workers',default=1)
    parser.add_argument('--IMG_LOG', type=str, help='image log',default='processed_images_log.csv')
    parser.add_argument('--HIDDEN_DIM', type=int, help='hidden dimension',default=512)
    parser.add_argument('--PATCH_SIZE', type=int, help='patch size',default=64)
    parser.add_argument('--IMAGE_SIZE', type=int, help='image size',default=512)
    parser.add_argument('--VAL_SIZE', type=float, help='validation size',default=0.15)
    parser.add_argument('--TEST_SIZE', type=float, help='test size',default=0.15)
    parser.add_argument('--MODEL_NAME', type=str, help='model name',default='lstm_base')
    parser.add_argument('--AE_MODEL_NAME', type=str, help='auto-encoder model name',default='CAE')
    parser.add_argument('--AE_PARAMETER_PATH', type=str, help='pretrained auto encoder parameter path',default='/content/Trained_Models/CAE_m_2024-10-13.pt')
    parser.add_argument('--ACCUMULATION_STEPS', type=int, help='accumulation steps',default=3)
    parser.add_argument('--SEQ_LEN', type=int, help='sequence length', default=16)
    parser.add_argument('--PRED_LEN', type=int, help='prediction length', default=4)
    parser.add_argument('--PAST_LEN', type=int, help='past length', default=10)
    

    args = parser.parse_args()


    EPOCHS = args.EPOCHS if args.EPOCHS else 1
    LR = args.LR if args.LR else 0.0001
    BATCH_SIZE = args.BATCH_SIZE if args.BATCH_SIZE else 2
    NUM_WORKERS = args.NUM_WORKERS if args.NUM_WORKERS else 1
    IMG_LOG = args.IMG_LOG if args.IMG_LOG else 'processed_images_log.csv'
    DATA_DIR = args.DATA_DIR if args.DATA_DIR else 'extracted_data'
    PATCH_SIZE = args.PATCH_SIZE if args.PATCH_SIZE else 64
    IMAGE_SIZE = args.IMAGE_SIZE if args.IMAGE_SIZE else 512
    HIDDEN_DIM = args.HIDDEN_DIM if args.HIDDEN_DIM else 512
    VAL_SIZE = args.VAL_SIZE if args.VAL_SIZE else 0.15
    TEST_SIZE = args.TEST_SIZE if args.TEST_SIZE else 0.15
    MODEL_NAME = args.MODEL_NAME if args.MODEL_NAME else 'lstm_base'
    AE_MODEL_NAME = args.AE_MODEL_NAME if args.AE_MODEL_NAME else 'CAE'
    AE_PARAMETER_PATH = args.AE_PARAMETER_PATH if args.AE_PARAMETER_PATH else '/content/Trained_Models/CAE_m_2024-10-13.pt'
    ACCUMULATION_STEPS = args.ACCUMULATION_STEPS if args.ACCUMULATION_STEPS else 3
    SEQ_LEN = args.SEQ_LEN if args.SEQ_LEN else 16
    PRED_LEN = args.PRED_LEN if args.PRED_LEN else 4
    PAST_LEN = args.PAST_LEN if args.PAST_LEN else 10

    # Load the data
    csv_file = IMG_LOG
    data_dir = DATA_DIR
    patch_size = PATCH_SIZE
    image_size = IMAGE_SIZE
    batch_size = BATCH_SIZE
    val_size = VAL_SIZE
    test_size = TEST_SIZE

    #AUTO ENCODER
    encoder_model = basemodels[AE_MODEL_NAME]['model'](HIDDEN_DIM)
    encoder_model.load_state_dict(torch.load(AE_PARAMETER_PATH))
    encoder_model.to(device)
    basemodels['CAE']['parameter_path']=AE_PARAMETER_PATH

    _, c, ps, ps = encoder_model.encoder(torch.rand(BATCH_SIZE*SEQ_LEN,1,PATCH_SIZE,PATCH_SIZE).to(device)).shape
    input_size = HIDDEN_DIM

    basemodel = basemodels[MODEL_NAME]
    model_param = [HIDDEN_DIM , 1,  HIDDEN_DIM]
    if MODEL_NAME == 'tft':
        model_param = [HIDDEN_DIM , 256 ,  HIDDEN_DIM]
        model = Combine_transformer_model(basemodel,basemodels[AE_MODEL_NAME],\
        input_size=input_size,model_param=model_param,model_name=MODEL_NAME, \
        hidden_dim=HIDDEN_DIM,pred_size=PRED_LEN ,sequence_length=SEQ_LEN)

    elif MODEL_NAME == 'patchTST':
        model_param = {
            'config': {
                'input_size': HIDDEN_DIM,
                'context_length': 10,
                'prediction_length': PRED_LEN,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 3,
                'num_input_channels': HIDDEN_DIM,
            }
        }
        model = Combine_transformer_model(basemodel,basemodels[AE_MODEL_NAME],\
        input_size=input_size,model_param=model_param,model_name=MODEL_NAME, \
        hidden_dim=HIDDEN_DIM,pred_size=PRED_LEN ,sequence_length=SEQ_LEN)

    else:
        model = Combine_model(basemodel,basemodels[AE_MODEL_NAME],\
        input_size=input_size ,model_param=model_param, pred_size=PRED_LEN ,\
        hidden_dim=HIDDEN_DIM,sequence_length=SEQ_LEN)

    model.to(device)
    optimizer = optim.Adam

    # Create the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders_2(csv_file, data_dir, NDVIDataset, 
            batch_size=batch_size, patch_size=patch_size, image_size=image_size, val_size=val_size, test_size=test_size,
            sequence_len=PAST_LEN,pred_len=PRED_LEN)
    
    #Early stopper
    early_stopping = EarlyStopping(tolerance=4, min_delta=5)
    
    history,basemodel = fit(EPOCHS, LR, model, train_dataloader,val_dataloader,\
                                 optimizer,accumulation_steps=ACCUMULATION_STEPS, \
                                 early_stopping=early_stopping)
    
    torch.save(basemodel.state_dict(), f'{MODEL_NAME}_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    # basemodel.save_weights(f'{MODEL_NAME}_encoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth', f'{MODEL_NAME}_decoder_weights{datetime.now().strftime("%Y-%m-%d")}.pth')
    np.save(f'{MODEL_NAME}_basemodelhistory{datetime.now().strftime("%Y-%m-%d")}.npy', history,allow_pickle=True)