
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
device = torch.device('cuda:1')
import random
from torch.autograd import Variable
# from utils import *
""" CNN Model """


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(CNN_RUL, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        self.flat = nn.Flatten()
        self.adapter = nn.Linear(14, 64) 
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            ) 
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.input_dim, self.input_dim, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            )
        self.regressor= nn.Sequential(
            nn.Linear(64, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1) )    
    def forward(self, src):
        features = self.encoder(src)
        reconstruction = self.decoder(features)
        features = self.maxpool(features)
        #print(self.flat(features).shape)
        predictions = self.regressor(self.adapter(self.flat(features)))
        return predictions, features, reconstruction


""" LSTM Model """


class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        #self.decoder = nn.LSTM(2*hidden_dim, int(input_dim/2), n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        #self.adapter = nn.Linear(self.hidden_size, input_dim)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
    def forward(self, src):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        #print(encoder_outputs.shape, hidden.shape)
        #reconstruction, (_, _) = self.decoder(encoder_outputs)

        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions, features, encoder_outputs
# model=LSTM_RUL(14, 32, 5, 0.5, True, device)

class LSTM_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        self.decoder = nn.LSTM(2*hidden_dim, int(input_dim/2), n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)

    def forward(self, encoder_outputs):
        reconstruction, (_, _) = self.decoder(encoder_outputs)
        
        return reconstruction



class Discriminator(nn.Module):
    def __init__(self, hidden_dims, bid):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dims*2, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
            nn.LogSoftmax())
    
    def forward(self, input):
        out = self.layer(input)
        return out



class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=6)
        )

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        #fea_2 = self.encoder_2(src)
        #fea_3 = self.encoder_3(src)
        
        return fea_1#, fea_2, fea_3


class CNN_RUL2(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = Encoder(input_dim=input_dim)
        #self.decoder = Decoder(input_dim=input_dim)
        self.flat = nn.Flatten()
        self.adapter = nn.Linear(14, 64)  # output = teacher network feature output
        self.regressor= nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        fea_1 = self.encoder(src)
        #self.decoder(fea_1, fea_2, fea_3)
        #features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        features = self.flat(fea_1)
        #hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, features #hint
        
def weights_init(m):
    for child in m.children():
        if isinstance(child,nn.Linear) or isinstance(child,nn.Conv1d):
            torch.nn.init.xavier_uniform_(child.weight)
