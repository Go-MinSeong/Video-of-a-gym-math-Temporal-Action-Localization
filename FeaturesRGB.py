# pip install natsort
# wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
# python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
# pip install cupy-cuda111
# pip install pydub
from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from FeaturesRGB_utils import run
from resnet import i3_res50
import os
from tqdm import tqdm

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
    #Path(outputpath).mkdir(parents=True, exist_ok=True)
    input_lst = os.listdir(datasetpath)
    video_lst = set([x.split('.')[0] for x in os.listdir(outputpath)])
    be_video_lst = set([x.split('.')[0] for x in input_lst])
    video_lst = sorted(list(be_video_lst - video_lst))
    vd_lst = [x for x in input_lst if x.split('.')[0] in video_lst]
    vd_lst = ["Fv_MFMVsHbU"]
    print(len(vd_lst))
    print(vd_lst)
    #video_lst = video_lst[:10]
    # setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    for video in tqdm(vd_lst):
        startime = time.time()
        video_path = datasetpath + "/" + video  # 이미지가 모여있는 경로를 설정

        print("Generating for {0}".format(video))

        features = run(i3d, frequency, video_path, batch_size, sample_mode)
        print(np.shape(features))
        features = np.squeeze(features)
        # 이미지 리스트를 받아와서 피쳐로 변환하는 코드 
        np.save(outputpath + "/" + video, features)
    
        # 피쳐 저장 코드

        print("Preprocessing done..")
        print("Obtained features of size: ", features.shape)

        #. shutil.rmtree(temppath) 해당 폴더 내 파일 삭제
        
        print("done in {0}.".format(time.time() - startime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str,
                        default="/home/work/TAL_FineGym/Data_img")
    parser.add_argument('--outputpath', type=str,
                        default="/home/work/TAL_FineGym/Feature_RGB")
    parser.add_argument('--pretrainedpath', type=str,
                        default="/home/work/TAL_FineGym/pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_mode', type=str, default="center_crop")
    args = parser.parse_args()
    generate(args.datasetpath, str(args.outputpath), args.pretrainedpath,
             args.frequency, args.batch_size, args.sample_mode)
