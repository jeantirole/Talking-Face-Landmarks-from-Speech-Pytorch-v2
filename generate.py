import tensorflow as tf
import librosa
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import os, shutil, subprocess
from keras import backend as K
from keras.models import Model, Sequential, load_model
from tqdm import tqdm
import utils
import argparse

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:] # (375,128)
    filler = melSpc[0, :] # (128)
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :] # (375,128)
        ctx = np.append(ctx, melSpc, axis=1) # (375,256) <- (375,128) + (375,128)
                                             # (375,384) <- (375,256) + (375,128)
                                             # (375,512) <- (375,384) + (375,128)
    return ctx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="input speech file")
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames, where each frame is 40 ms")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

output_path = args.out_fold
num_features_Y = 136
num_frames = 75
wsize = 0.04
hsize = wsize
fs = 44100
trainDelay = args.delay
ctxWin = args.ctx

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

model = load_model(args.model)
model.summary()

test_file = args.in_file

# Used for padding zeros to first and second temporal differences
zeroVecD = np.zeros((1, 64), dtype='f16')
zeroVecDD = np.zeros((2, 64), dtype='f16')

# Load speech and extract features
sound, sr = librosa.load(test_file, sr=fs) # (661280,) mono channel
melFrames = np.transpose(utils.melSpectra(sound, sr, wsize, hsize)) # (375,64)
melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0) # (375,64)
melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0) # (375,64)

features = np.concatenate((melDelta, melDDelta), axis=1) # (375,128) # 같은 길이에 double dimension 

'''
At each time step, the
input to the network is the first and second order temporal differences of the log mel spectra of the current and the previous N frames. This provides short-term contextual information
'''
features = addContext(features, ctxWin) # (375, 512) # 128 을 좀 겹쳐서 128 => 512 로 dimension 이 늘어났다. 
features = np.reshape(features, (1, features.shape[0], features.shape[1])) # (1, 375, 512) 로 batch dim 을 위해서 reshape

upper_limit = features.shape[1] # 375
lower = 0 
generated = np.zeros((0, num_features_Y)) # num_features_Y = 136 # 미리 정해진 값 # (1,136)

# Generates face landmarks one-by-one
# This part can be modified to predict the whole sequence at one, but may introduce discontinuities
for i in tqdm(range(upper_limit)): # 375 개의 mel_frames 들이 있음. 각각의 frame 에 대해서 추론시작 
    cur_features = np.zeros((1, num_frames, features.shape[2])) # (1,75,512) 75 <-num_frames 추론할 frame 크기. 
    if i+1 > 75:
        lower = i+1-75
    cur_features[:,-i-1:,:] = features[:,lower:i+1,:] # (1,75,512)

    #-------------------------------------
    # model prediction 
    pred = model.predict(cur_features) # pred (1,75,136) # 136<- lstm 의 last_hidden_vector size 이기도 하면서, 68차원x2 가 필요한 (x,y) 
    generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0) # (374, 136) # 아까 frame 375 개에 대해서 각 136 차원 값   

# Shift the array to remove the delay
generated = generated[trainDelay:, :] # (374,136)
tmp = generated[-1:, :] # (1,136)
for _ in range(trainDelay):
    generated = np.append(generated, tmp, axis=0)

if len(generated.shape) < 3:
    # float => int debugged by eric 
    generated = np.reshape(generated, (int(generated.shape[0]), int(generated.shape[1]/2), 2))
    # (375, 68, 2)

fnorm = utils.faceNormalizer()
generated = fnorm.alignEyePointsV2(600*generated) / 600.0 
utils.write_video_wpts_wsound(generated, sound, fs, output_path, 'PD_pts', [0, 1], [0, 1])


