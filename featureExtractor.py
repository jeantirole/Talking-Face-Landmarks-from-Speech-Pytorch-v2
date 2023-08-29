# Written by S. Emre Eskimez, in 2017 - University of Rochester
# This script is written for extracting features from GRID dataset. 
# If you intend to use other videos with arbitrary length, you need to modify this script.
# Usage: python featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file-name.hdf5
# You can find shape_predictor_68_face_landmarks.dat online from various sources.
 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from copy import deepcopy
import sys
import os
import dlib
import glob
# from skimage import io
import numpy as np
import h5py
import pylab
import librosa
import imageio
import utils
import argparse, fnmatch, shutil
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-vp", "--video-path", default="/mnt/hdd/eric/.tmp_ipy_d/make/TalkingFaceGeneration_Pytorch/data/s1/video",type=str, help="video folder path")
parser.add_argument("-sp", "--sp-path", default="/mnt/hdd/eric/.tmp_ipy_d/make/TalkingFaceGeneration_Pytorch/data/shape_predictor_68_face_landmarks.dat", type=str, help="shape_predictor_68_face_landmarks.dat path")
parser.add_argument("-o", "--output-path", default="/mnt/hdd/eric/.tmp_ipy_d/make/TalkingFaceGeneration_Pytorch/output/eric_output_tmp.hdf5", type=str, help="Output file path")
args = parser.parse_args()

predictor_path = args.sp_path#'../data/shape_predictor_68_face_landmarks.dat'
video_folder_path = args.video_path
dataset_path = args.output_path

ms = np.load('mean_shape.npy') # Mean face shape, you can use any kind of face instead of mean face.
fnorm = utils.faceNormalizer()
ms = fnorm.alignEyePoints(np.reshape(ms, (1, 68, 2)))[0,:,:]

try:
    os.remove(dataset_path)
except:
    print ('Exception when deleting previous dataset...')

wsize = 0.04
hsize = 0.04

# These two vectors are for filling the empty cells with zeros for delta and double delta features
zeroVecD = np.zeros((1, 64))
zeroVecDD = np.zeros((2, 64))

# ----
# Eric
dataHandler = h5py.File(dataset_path,'w')

speechData = dataHandler.create_dataset('MelFeatures', (1, 75, 128), maxshape=(None, 75, 128)) # 
lmarkData = dataHandler.create_dataset('flmark', (1, 75, 136), maxshape=(None, 75, 136)) # 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

points_old = np.zeros((68, 2), dtype=np.float32)

fileCtr = 0

for root, dirnames, filenames in os.walk(video_folder_path):
    #print(filenames)
    for filename in filenames:
        # You can add the file type of your videos here:
        if os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mp4':
            #print(os.path.splitext(filename)[1])
            f = os.path.join(root, filename)
            print("f : ",f)
            vid = imageio.get_reader(f,  'ffmpeg')
            point_seq = []
            img_seq = []

            # Eric
            # debugged 
            # vid.count_frames()
            print("### count frames : ", vid.count_frames()) # Multiply the frame rate (in frames per second) by the playing time in seconds 40ms => 1 frame 
            for frm_cnt in tqdm(range(0, int(vid.count_frames()))):
                points = np.zeros((68, 2), dtype=np.float32)

                try:
                    img = vid.get_data(frm_cnt)
                except:
                    print('FRAME EXCEPTION!!')
                    continue

                dets = detector(img, 1)
                if len(dets) != 1:
                    print('FACE DETECTION FAILED!!')
                    continue

                for k, d in enumerate(dets):
                    shape = predictor(img, d)

                    for i in range(68):
                        points[i, 0] = shape.part(i).x
                        points[i, 1] = shape.part(i).y

                # points = np.reshape(points, (points.shape[0]*points.shape[1], ))
                point_seq.append(deepcopy(points))

            # ----
            # Speech Extraction from a video  
            cmd = 'ffmpeg -y -i '+os.path.join(root, filename)+' -vn -acodec pcm_s16le -ac 1 -ar 44100 temp.wav'
            subprocess.call(cmd, shell=True) 
            
            # load speech from wav file 
            y, sr = librosa.load('temp.wav', sr=44100) # y shape : (131328,)

            # cuz we already saved wav as "y" we don't need "temp.wav"
            os.remove('temp.wav')

            # we saved points of each frame at the "point_seq"
            frames = np.array(point_seq) # frames.shape => (75,68,2) 
            # 75 frame 인 이유 ! 
            # The videos use a frame rate of
            # 25 frames per second (FPS), resulting in 75 frames for each video.
            # face normalzier 
            fnorm = utils.faceNormalizer() 
            # frames => alignEyePoints
            aligned_frames = fnorm.alignEyePoints(frames)
            # transfer Experession? 
            transferredFrames = fnorm.transferExpression(aligned_frames, ms)
            frames = fnorm.unitNorm(transferredFrames)

            if frames.shape[0] != 75:
                continue
                
            # whole audios => mel-Spectrum  
            melFrames = np.transpose(utils.melSpectra(y, sr, wsize, hsize)) # (75,64) 
            #  mel-spectogram 에서 정해준 dimension 64 ! 
            #  We calculate 64 bin log-mel spectra of the speech
            #  signal covering the entire frequency range using a 40 ms hanning window without
            #  any overlap to match the video frame rate. 
            #  wsize = 0.04, hsize = 0.04

            # delta features 
            melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0) # (75,64) # mel-frames 를 그대로 쓰지 않고 한번 차분 
            melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0) # (75,64)
            
            melFeatures = np.concatenate((melDelta, melDDelta), axis=1) # (75,128)

            if melFeatures.shape[0] != 75:
                continue

            speechData[fileCtr, :, :] = melFeatures # (n,75,128) 
            speechData.resize((speechData.shape[0]+1, 75, 128)) # 한 프레임 당, 68 dimension 의 melData 가 *2개 붙어있는 구조. 

            lmarkData[fileCtr, :, :] = np.reshape(frames, (75, 136))
            lmarkData.resize((lmarkData.shape[0]+1, 75, 136)) #  한 frame 당, (1/75) 하나의 mask(68 * 2)

            fileCtr += 1