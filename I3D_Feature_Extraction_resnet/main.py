from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from models.resnet import i3_res50
import os
from ffmpegpython import ffmpeg
import sys
mps = torch.device("mps")
# sys.path.append(
#     '/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/ffmpeg-python/')

# bring ffmpeg git because error of ffmpeg and ffmpeg.input
# print(dir(ffmpeg))
# python main.py datasetpath "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/movie" --outputpath "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/features"
# conda install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
    video_name_lst = []
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath + "/temp/"
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob('**/*.mp4')]
    # setup the model     여기서는 ffmpeg 처리만 해보자
    #i3d = i3_res50(400, pretrainedpath)
    # i3d.to(mps)
    # i3d.train(False)  # Set model to evaluate mode
    for video in videos:  # 비오 파일을 하나씩 확ㅣ한다.
        # videoname = video.split("/")[-1].split(".")[0]
        videoname = video.split(".")[0]
        startime = time.time()
        print("Generating for {0}".format(video))
        #
        #print("checking", temppath)

        #print("checking", temppath)

        path = temppath+videoname.split('/')[-1] + "/"
        Path(path).mkdir(parents=True, exist_ok=True)  # 파일 생성 해주는 코드
        #print("checking", path)

        ffmpeg.input(video).output('{}%d.jpg'.format(path),
                                   start_number=0).global_args('-loglevel', 'quiet').run()
        video_name_lst.append(videoname)
        print("Preprocessing done..")
    print(video_name_lst)
    with open('video_name.pkl', 'wb') as f:
        pickle.dump(video_name_lst, f)
        # features = run(i3d, frequency, temppath, batch_size, sample_mode)  # feature extraction이 진행된다.
        #np.save(outputpath + "/" + videoname, features)
        #print("Obtained features of size: ", features.shape)
        # shutil.rmtree(temppath)
        #print("done in {0}.".format(time.time() - startime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str,
                        default="/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/movie")
    parser.add_argument('--outputpath', type=str,
                        default="/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/output_3")
    parser.add_argument('--pretrainedpath', type=str,
                        default="/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)  # 16
    parser.add_argument('--batch_size', type=int, default=20)  # 20
    parser.add_argument('--sample_mode', type=str, default="oversample")
    args = parser.parse_args()
    generate(args.datasetpath, str(args.outputpath), args.pretrainedpath,
             args.frequency, args.batch_size, args.sample_mode)
