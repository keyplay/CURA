import torch
import numpy as np 
device = torch.device('cuda')
import pandas as pd
from models.my_models import *
from models.models_config import get_model_config
#Different Domain Adaptation  approaches
import importlib
import random
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from models.models import get_backbone_class, Model
from utils import starting_logs
import argparse
parser = argparse.ArgumentParser()

print(torch.cuda.current_device())




if __name__ == '__main__':
  # ========= Select the DA methods ============
  parser.add_argument('--da_method', default='SFUDA', type=str, help='DDC, Deep_Coral, DANN_cons, ADARUL, CADA, Tri_unc_post_mmd, Tri_mmd')

  # ========= Select the DATASET ==============
  parser.add_argument('--dataset', default='CMAPSS', type=str, help='Dataset of choice: (CMAPSS - NCMAPSS)')

  # ========= Select the BACKBONE ==============
  parser.add_argument('--backbone', default='LSTM', type=str, help='Backbone of choice: (CNN - LSTM - TCN)')

  # ========= Experiment settings ===============
  parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')

  parser.add_argument('--exp_log_dir', default='./log', type=str, help='log file path')
  # arguments
  args = parser.parse_args()

  method = importlib.import_module(f'trainer.cross_domain_models.{args.da_method}')
  
  data_path= "./data/datasets/"+args.dataset
  
  dataset_class = get_dataset_class(args.dataset)
  hparams_class = get_hparams_class(args.dataset)
  dataset_configs = dataset_class()
  hparams_class = hparams_class()
  hparams = {**hparams_class.alg_hparams[args.da_method], **hparams_class.train_params}
 
  df=pd.DataFrame();res = [];full_res = []
  print('=' * 89)
  print (f'Domain Adaptation using: {args.da_method}')
  print('=' * 89)
  
  for src_id, tgt_id in dataset_configs.scenarios:
      total_loss, total_score = [], []
      total_best_loss, total_best_score = [], []

      seed = 42
      for run_id in range(args.num_runs):
          seed += 1
          torch.manual_seed(seed)
          np.random.seed(seed)
          random.seed(seed)
          
          logger, scenario_log_dir = starting_logs(args.dataset, args.da_method, args.exp_log_dir, src_id, tgt_id, run_id)

          src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score = method.cross_domain_train(device,args.dataset,dataset_configs, hparams, args.backbone, data_path, src_id,tgt_id,run_id, logger)
          
          total_loss.append(test_loss)
          total_score.append(test_score)
          total_best_loss.append(best_loss)
          total_best_score.append(best_score)
      loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
      score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
      best_loss_mean, best_loss_std = np.mean(np.array(total_best_loss)), np.std(np.array(total_best_loss))
      best_score_mean, best_score_std = np.mean(np.array(total_best_score)), np.std(np.array(total_best_score))
      full_res.append((f'run_id:{run_id}',f'{src_id}-->{tgt_id}', f'{src_only_loss:2.4f}' ,f'{loss_mean:2.4f}',f'{loss_std:2.4f}',f'{best_loss_mean:2.4f}',f'{best_loss_std:2.4f}',f'{src_only_score:2.4f}',f'{score_mean:2.4f}',f'{score_std:2.4f}',f'{best_score_mean:2.4f}',f'{best_score_std:2.4f}'))
              
  df= df.append(pd.Series((f'{args.da_method}')), ignore_index=True)
  df= df.append(pd.Series(("run_id", 'scenario','src_only_loss', 'mean_loss','std_loss', 'best_mean_loss','best_std_loss', 'src_only_score', f'mean_score',f'std_score', f'best_mean_score',f'best_std_score')), ignore_index=True)
  df = df.append(pd.DataFrame(full_res), ignore_index=True)
  print('=' * 89)
  print (f'Results using: {args.da_method}')
  print('=' * 89)
  print(df.to_string())
  df.to_csv(f'./results/results_complete_{args.dataset}_{args.da_method}_{args.backbone}_evi_{dataset_configs.evidential}_du.csv')
