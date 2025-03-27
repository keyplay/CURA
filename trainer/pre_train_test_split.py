##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./opts.py stores the options
# The file ./train_eval.py stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
import sys
sys.path.append("..")
import warnings
import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from trainer.train_eval import train, evaluate

# fix_randomness(5)
device = torch.device('cuda')


def pre_train(model, train_dl, test_dl, data_id, config, params):
    # criteierion
      
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(params['pretrain_epochs']):
        start_time = time.time()
        train_loss, train_score, train_feat, train_labels = train(model, train_dl, optimizer, criterion, config, device)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')
        # Evaluate on the test set
        if epoch % 5 == 0:
            test_loss, test_score, test_nll, _, _,_,_ = evaluate(model, test_dl, criterion, config, device)
            print('=' * 89)
            print(f'\t  Performance on test set::: Loss: {test_loss:.3f} | Score: {test_score:7.3f} | Nll: {test_nll:.3f}')
    
    
    # Evaluate on the test set
    test_loss, test_score, test_nll, _, _, predicted_rul, _ = evaluate(model, test_dl, criterion, config, device)
    
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f} | Nll: {test_nll:.3f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return model


