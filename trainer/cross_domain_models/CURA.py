import sys

sys.path.append("..")
from utils import *
from data.mydataset import data_generator
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import copy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
from trainer.train_eval import train
from models.models import get_backbone_class, Model
from utils import evidential_unreason_loss
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.manifold import TSNE

def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):
    
    src_train_dl = data_generator(data_path, src_id, dataset_configs, hparams, "train")
    tgt_train_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "train")
    tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")
    src_test_dl = data_generator(data_path, src_id, dataset_configs, hparams, "test")

    logger.info('Restore source pre_trained model...')
    if dataset_configs.evidential=='unreason':
        checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_EVI_unreason_{src_id}.pt')
    else:
        checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_{src_id}.pt')

    # pretrained source model
    logger.info('=' * 89)
    
    source_model = Model(dataset_configs, backbone).to(device) 
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    # initialize target model
    target_model = Model(dataset_configs, backbone).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    target_encoder = target_model.feature_extractor
    target_encoder.train()
    set_requires_grad(target_encoder, requires_grad=True)
    set_requires_grad(target_model.regressor, requires_grad=False)

    all_source_var = []
    for step, (source_x, source_y) in enumerate(src_train_dl):
        source_x = source_x.to(device) 
        source_output, source_features = source_model(source_x)
        mu, v, alpha, beta = source_output      
             
        var = torch.sqrt(1/v)
        
        all_source_var.append(var.detach())

            
    all_source_var = torch.cat(all_source_var, dim=0)   
    src_model_unc = torch.mean(all_source_var)

    print('model_unc', src_model_unc)
  
    # optimizer
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))   
    
    best_score, best_loss = 1e10, 1e10 
    unc_threshold = hparams['init_unc_threshold'] 
    for epoch in range(1, hparams['num_epochs'] + 1):
        total_loss = 0
        total_var, total_sigma = 0, 0
        start_time = time.time()
        
        if epoch % hparams['threshold_update_interval'] == 0 and unc_threshold < hparams['max_unc_threshold']:
            unc_threshold += hparams['unc_threshold_step'] 
        
        num_select = 0              
        for step, (target_x, target_y) in enumerate(tgt_train_dl):

            target_optim.zero_grad()

            target_x = target_x.to(device) 
            target_output, target_features = target_model(target_x)
            
            mu, v, alpha, beta = target_output
            
            var = torch.sqrt(1/v)
            sigma = torch.sqrt(beta*(1+v.detach())/(alpha*v.detach()))
                     
            
            select_idx = torch.abs(var-src_model_unc)>src_model_unc*unc_threshold
            num_select += torch.sum(select_idx)
            
            total_var += torch.mean(var)
            total_sigma += torch.mean(sigma)
            
            loss = torch.abs((src_model_unc-torch.mean(var[select_idx])))
           
            loss.backward()
            target_optim.step()
            total_loss += loss.item()
        
             
        mean_loss = total_loss / (step+1)
        mean_var = total_var / (step+1)
        mean_sigma = total_sigma / (step+1)        
        logger.info(f'Epoch: {epoch:02}')
        logger.info(f'target_loss:{mean_loss}, mean_var:{mean_var}, mean_sigma:{mean_sigma}')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, src_only_nll, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
            test_loss, test_score, test_nll, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
            if best_score > test_score:
                best_loss, best_score = test_loss, test_score
                
            logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score} \t Src_Only Nll:{src_only_nll}')
            logger.info(f'DA RMSE:{test_loss} \t DA Score:{test_score} \t DA Nll:{test_nll}')
            
        print("num_select", num_select / len(tgt_train_dl.dataset))
        if num_select / len(tgt_train_dl.dataset) < hparams['stop_ratio'] : break       
            
    src_only_loss, src_only_score, src_only_nll, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
    test_loss, test_score, test_nll, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
       
    
    logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score} \t Src_Only Nll:{src_only_nll}')
    logger.info(f'After DA RMSE:{test_loss} \t After DA Score:{test_score} \t DA Nll:{test_nll}')
    
    return src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score
