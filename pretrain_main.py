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
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from data.mydataset import data_generator
from models.models import get_backbone_class, Model
import os
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


backbone = 'LSTM'


select_method = 'NO_ADAPT'
dataset = 'CMAPSS'
data_path= "./data/datasets/"+dataset
directory = f'./trained_models/{dataset}/single_domain_dr05'

dataset_class = get_dataset_class(dataset)
hparams_class = get_hparams_class(dataset)
dataset_configs = dataset_class()
hparams_class = hparams_class()
hparams = {**hparams_class.alg_hparams[select_method], **hparams_class.train_params}

def main():
    df=pd.DataFrame()
    res = []

    full_res = []
    unique_src_id = {src_id for src_id, tgt_id in dataset_configs.scenarios}
    for src_id in unique_src_id:
        print('Initializing model for', src_id)

        source_model = Model(dataset_configs, backbone).to(device)
        print(source_model)
        #source_model.apply(weights_init)
        print('=' * 89)
        print(f'The {backbone} has {count_parameters(source_model):,} trainable parameters')
        print('=' * 89)
        print('Load_source_target datasets...')

        src_train_dl = data_generator(data_path, src_id, dataset_configs, hparams, "train")
        src_test_dl = data_generator(data_path, src_id, dataset_configs, hparams, "test")  
            
        trained_source_model = pre_train(source_model, src_train_dl, src_test_dl, src_id, dataset_configs, hparams)
        # saving last epoch model
        if hparams['save']:
            checkpoint1 = {'state_dict': trained_source_model.state_dict(),}
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            if dataset_configs.evidential=='quantile':
                torch.save(checkpoint1, directory+f'/pretrained_{backbone}_EVI_quant_{src_id}.pt')
            elif dataset_configs.evidential=='org':
                torch.save(checkpoint1, directory+f'/pretrained_{backbone}_EVI_org_{src_id}.pt')
            elif dataset_configs.evidential=='unreason':
                torch.save(checkpoint1, directory+f'/pretrained_{backbone}_EVI_unreason_{src_id}.pt')
            else:
                torch.save(checkpoint1, directory+f'/pretrained_{backbone}_{src_id}2.pt')
                       
        # test on cross domain data
        for tgt_id in unique_src_id:
            if src_id != tgt_id:
                tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test") 
                criterion = RMSELoss()
                test_loss, test_score, test_nll, _, _, _, _ = evaluate(trained_source_model, tgt_test_dl, criterion, dataset_configs, device)
                print('=' * 89)
                print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f} | Nll: {test_nll:.3f}')
                print('=' * 89)  


main()
print('Finished')
