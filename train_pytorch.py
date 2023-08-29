import tensorflow as tf
import librosa
import numpy as np
import os, shutil, subprocess
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Reshape, Activation, Dropout, Flatten
from keras.models import Model
from tqdm import tqdm
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,Adam
#from keras.optimizers import RMSprop, Adam
import h5py
from keras.callbacks import TensorBoard
import argparse, fnmatch
import pickle
import random
import time, datetime


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", default="/mnt/hdd/eric/.tmp_ipy_d/make/TalkingFaceGeneration_Pytorch/output/eric_output.hdf5",type=str, help="Input file containing train data")
parser.add_argument("-u", "--hid-unit", default=512, type=int, help="hidden units")

# The amount of delay we introduce
# is between 1 (40 ms) and 5 frames (200 ms).
parser.add_argument("-d", "--delay", default=1,type=int, help="Delay in terms of number of frames")

parser.add_argument("-c", "--ctx", default=3,type=int, help="context window size")
# can find "3" in generator.py

parser.add_argument("-o", "--out-fold", default="/mnt/hdd/eric/.tmp_ipy_d/make/TalkingFaceGeneration_Pytorch/train_output",type=str, help="output folder")
args = parser.parse_args()




#--- 
# output folder creation 
output_path = args.out_fold+'_'+str(args.hid_unit)+'/'

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)


#---
# arguments
ctxWin = args.ctx
num_features_X = 128 * (ctxWin+1)# input feature size # 128 * 3 = 512
num_features_Y = 136 # output feature size --> (68, 2)
num_frames = 75 # time-steps
batchsize = 128
h_dim = args.hid_unit
lr = 1e-3
drpRate = 0.2 # Dropout rate 
recDrpRate = 0.2 # Recurrent Dropout rate 
frameDelay = args.delay # Time delay
numEpochs = 200

#---
# Data
dset = h5py.File(args.in_file, 'r')
numIt = int(dset['flmark'].shape[0]//batchsize) + 1 # doens't need cuz i changed to torch ! 
metrics = ['MSE', 'MAE']


#---
# Custom Torch Dataset Pipe
from torch.utils.data import Dataset, DataLoader
import torch 

class Custom_Dataset(Dataset):
    ''' torch version data pipeline '''
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['MelFeatures'])
    
    
    def __getitem__(self,idx):
        cur_lmark = self.data['flmark'][idx, :, :]
        cur_mel = self.data['MelFeatures'][idx, :, :]
        
        def addContext(melSpc, ctxWin):
            ctx = melSpc[:,:]
            filler = melSpc[0, :]
            for i in range(ctxWin):
                melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
                ctx = np.append(ctx, melSpc, axis=1)
            return ctx

        if frameDelay > 0:
            filler = np.tile(cur_lmark[0:1, :], [frameDelay, 1])
            cur_lmark = np.insert(cur_lmark, 0, filler, axis=0)[:num_frames]
        
        X = addContext(cur_mel, ctxWin)
        Y_= cur_lmark

        out = {'cur_mel':torch.from_numpy(X), 'cur_lmark':torch.from_numpy(Y_)}
        return out  
    

#--- 
# build torch model 
'''
input_dim = 512
hidden_dim = 512 
output_dim = 136
sequence length doesn't matter, torch will take it be considered
'''
import torch.nn as nn

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=3, batch_first=True)
        self.lstm_4 = nn.LSTM(input_size=512, hidden_size=136, num_layers=1,batch_first=True)
        
    def forward(self, x):
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_4(x)
        return x


#---
# Train Pipes (Scratch)

# MSE and adam 
import torch.optim as optim
device = "cuda:0"

# model
model = AirModel()
model.to(device)

# optimizer & loss
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

# dataset
lm_dataset = Custom_Dataset(data=dset)
loader = DataLoader(lm_dataset, batch_size=128, 
                        shuffle=True, num_workers=4)

# training pipes 
n_epochs = 200 
for epoch in range(n_epochs):
    model.train()
    
    # train
    for i,data_ in enumerate(loader):
        x_train = data_["cur_mel"].to(device)
        y_train = data_["cur_lmark"].to(device)
        
        y_pred = model(x_train)
        loss   = loss_fn(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 ==0:
            print(f"{np.sqrt(loss.detach().cpu())}")

    # model save 
    if epoch % 10 ==0:
        torch.save(model.state_dict(), os.path.join( args.out_fold, f"{epoch}_model.pth"))
        
        
    