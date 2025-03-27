##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./trainer/ stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
# imports
import time

import torch
from torch import nn
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import pandas as pd
import argparse
from tqdm import tqdm, trange
import warnings
from utils import *
from models.my_models import *
from trainer.train_eval import train, evaluate
from trainer.pre_train_test_split import pre_train
from sklearn.exceptions import DataConversionWarning
from models.models_config import get_model_config, initlize
from data.mydataset import create_dataset, create_dataset_full
# torch.nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Steps
# Step 1: Load Dataset
# Step 2: Create Model Class
# Step 3: Load Model
# Step 2: Make Dataset Iterable
# Step 4: Instantiate Model Class
# Step 5: Instantiate Loss Class
# Step 6: Instantiate Optimizer Class
# Step 7: Train Mode

"""Configureations"""
params = {'window_length': 30, 'sequence_length': 30, 'batch_size': 256, 'input_dim': 14, 'src_pretrain': False, 'pretrain_epoch': 30, 'save': True,
          'data_path': r"./data/processed_data/cmapps_train_test_cross_domain.pt", "data_type": 'log',
          'dropout': 0.5,  'lr': 1e-3, 'tensorboard':False}


# load data
my_dataset = torch.load(params['data_path'])
# load model
config = get_model_config('LSTM') #CNN_AE



def main():
    df=pd.DataFrame()
    res = []
    # pm = Symbol(u'±')
    full_res = []
    for src_id in ['FD001', 'FD002', 'FD003','FD004']:
        print('Initializing model for', src_id)
        #source_model = CNN_RUL(params['input_dim'], 32, 0.5).to(device)
        #source_model.apply(weights_init)
        print('=' * 89)
        print('Load_source_target datasets...')
        
        # test on cross domain data
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            if src_id != tgt_id:
                total_loss = []
                total_score = []
                
                tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id])
                
                for run_id in range(5):
                    tgt_model = LSTM_RUL(14, 32, 5, 0.5, True, device).to(device)
                    da_tgt_checkpoint = torch.load(f'./trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_0.1_nD_tgt.pt')
                    tgt_model.load_state_dict(da_tgt_checkpoint)
                
                    criterion = RMSELoss()
                    test_loss, test_score, _, _, _, _ = evaluate(tgt_model, tgt_test_dl, criterion, config, device)
                    print(test_loss, test_score)
                    total_loss.append(test_loss)
                    total_score.append(test_score)

                loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
                score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
                full_res.append((f'run_id:{run_id}',f'{src_id}-->{tgt_id}' ,f'{loss_mean:2.4f}',f'{loss_std:2.4f}',f'{score_mean:2.4f}',f'{score_std:2.4f}'))
    
    df = df.append(pd.DataFrame(full_res), ignore_index=True)
    print('=' * 89)
    print(df.to_string())



main()
print('Finished')
print('Finished')
