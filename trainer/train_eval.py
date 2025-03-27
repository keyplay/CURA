import torch
from utils import scoring_func, quant_evi_loss, evidential_loss, evidential_unreason_loss, evidential_unreason_loss2, NIG_NLL_org

def train(model, train_dl, optimizer, criterion, config,device):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    for inputs, labels in train_dl:
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        output, feat = model(src)
        rul_loss = 0
        if config.evidential=='quantile':
            mu, v, alpha, beta = output
            pred = torch.mean(mu.detach(), dim=1)
            for i, q in enumerate(config.quantiles):
                rul_loss += quant_evi_loss(labels.unsqueeze(-1), mu[:, i].unsqueeze(-1), v[:, i].unsqueeze(-1),
                               alpha[:, i].unsqueeze(-1), beta[:, i].unsqueeze(-1), q, coeff=3e-1)
        elif config.evidential=='org':
            mu, v, alpha, beta = output
            pred = mu
            rul_loss = evidential_loss(labels.unsqueeze(-1), mu, v, alpha, beta, coeff=1)
            #rul_loss = evidential_loss(labels, mu, v, alpha, beta, coeff=3e-1)
            #print(mu[:5], v[:5], alpha[:5], beta[:5], rul_loss)
        elif config.evidential=='unreason':
            mu, v, alpha, beta = output
            pred = mu
            #print(labels.shape, mu.shape, v.shape, alpha.shape, beta.shape)
            rul_loss = evidential_unreason_loss(labels, mu.squeeze(), v.squeeze(), alpha.squeeze(), beta.squeeze(), coeff=1)
        else:
            pred = output
            rul_loss = criterion(pred.squeeze(), labels)
        
        #denormalization
        pred  = pred * config.max_rul
        labels = labels * config.max_rul
        #print(pred.shape, labels.shape)
        score = scoring_func(pred.squeeze() - labels)

        rul_loss.backward()
        if (type(model.feature_extractor).__name__=='LSTM'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score
    return epoch_loss / len(train_dl), epoch_score, pred, labels

def evaluate_MCD(model, regressor2, test_dl, criterion, config, device, denorm_flag=True):
    model.eval()
    regressor2.eval()
    total_feas=[];total_labels=[]
    epoch_loss = 0
    epoch_nll = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    var_list = []
    sigma_list = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)          
            labels = labels.to(device)

            pred1, feat = model(src)
            pred2 = regressor2(feat)
            pred = (pred1 + pred2) / 2.0
                
            # denormalize predictions
            if denorm_flag:
                pred = pred * config.max_rul
                if labels.max() <= 1:
                    labels = labels * config.max_rul
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()

    model.train()
    return epoch_loss / len(test_dl), epoch_score, epoch_nll / len(test_dl),torch.cat(total_feas), [var_list, sigma_list], predicted_rul, true_labels
    

def evaluate(model, test_dl, criterion, config, device, denorm_flag=True):
    model.eval()
    total_feas=[];total_labels=[]
    epoch_loss = 0
    epoch_nll = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    var_list = []
    sigma_list = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)          
            labels = labels.to(device)

            test_output, feat = model(src)
            if config.evidential=='quantile':
                mu, v, alpha, beta = test_output
                var = torch.sqrt((beta /(v*(alpha - 1))))
                sigma = beta/(alpha-1.0)
                pred = torch.mean(mu, dim=1)
                var_list.append(var)
                sigma_list.append(sigma)
            elif config.evidential in ['org', 'unreason']:
                mu, v, alpha, beta = test_output
                var = torch.sqrt((beta /(v*(alpha - 1))))
                sigma = beta/(alpha-1.0)
                pred = mu
                var_list.append(var)
                sigma_list.append(sigma)
            else:
                pred = test_output
                
            # denormalize predictions
            if denorm_flag:
                pred = pred * config.max_rul
                if labels.max() <= 1:
                    labels = labels * config.max_rul
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            
            if config.evidential in ['org', 'unreason']:
                nll = NIG_NLL_org(labels.unsqueeze(-1), pred, v, alpha, beta)
                epoch_nll += nll.item()
            
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()
    
    if len(var_list) > 0:
        var_list = torch.cat(var_list, dim=0)
        sigma_list = torch.cat(sigma_list, dim=0)

    model.train()
    return epoch_loss / len(test_dl), epoch_score, epoch_nll / len(test_dl),torch.cat(total_feas), [var_list, sigma_list], predicted_rul, true_labels
