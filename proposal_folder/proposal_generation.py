# 23개가 74만개
import numpy as np 
import pandas as pd
from classifier import ConvNet
import torch
import json
from utils import right_proposal, video_info
import os
from tqdm import tqdm
import random
from utils import right_proposal, cal_iou, save_window
import pickle
import random
import time
# train :  213 , validation :  55 으로 스플릿 진행합니다.
def continuous_random(x, y):
    num=random.randrange(x,y-7)
    return num,num+8

def main(dir_home = "/home/work/TAL_FineGym/Feature_RGB", only="1", thresh_hold=0.05):
    # sliding window를 거치면서 분할된 window들을 저장할 리스트
    tresult_lst=[]
    vresult_lst=[]
    tresultL_lst=[]
    vresultL_lst=[]
    features = os.listdir(dir_home)
    random.seed(0)

    # json 파일 불러오기
    with open('/home/work/TAL_FineGym/Label/finegym_annotation.json', 'r') as f:
        data = json.load(f)

    train_rate = int(len(features)*0.8)
    # 120 , 31
    print("train : ",train_rate,", validation : " ,len(features)-train_rate, "으로 스플릿 진행합니다.")
    #features= features[::-1]
    for k, f in tqdm(enumerate(features)):
        subset = data["databases"][f.split(".")[0]]["subset"]
        if subset=="val" and only=="1": # only가 1일 경우 train proposal만 생성
            print("continue");
            continue;
        if subset=="train" and only=="2": # only가 2일 경우 val proposal만 생성
            print("continue");
            continue;
        window_list = []
        vector = np.load(dir_home+"/"+f)
        fps = video_info(f.split(".")[0])["fps"]
        lst = data["databases"][f.split(".")[0]]["annotation"]
        
        # 기록
        res = open("result.txt",'a')
        res.write("%d번째 줄입니다.\n" % k)
        res.write(f + "\n")
        res.write(str(vector.shape)+"\n")
        res.write(str(time.strftime('%Y.%m.%d - %H:%M:%S'))+"\n")
        res.close()

        lst_1 = [[ y*fps//16 for y in x["segment"]] for x in lst]
        print(f)
        print(vector.shape)
        
        window_lengths = [8, 16, 32, 64, 128, 256, 512]  # sliding window의 길이
        # 1당 0.5 ~ 0.7 초 가량을 의미함 -> 8 = 4 ~ 5.6초

        # 정답 어노테이션에서 추출하는 프로포설
        for length in window_lengths:
            stride = int(length * 0.25)  # stride는 window length의 25%
            for l in lst_1: # l [12, 23]
                l_0, l_1=int(l[0]), int(l[1])
                if l_1-l_0 < length:
                    continue;
                num_windows = (l_1-l_0 - length) // stride + 1

                for i in range(num_windows):
                    random_value = random.random()
                    if random_value >= thresh_hold:
                        continue
                    if length == 8:
                        v = vector[l_0+i*stride:l_0+i*stride+length, :]
                        if np.shape(v)!=(8, 2048):
                            continue;
                        window_list.append((v, [l_0+i*stride,l_0+i*stride+length], f.split(".")[0]))
                    else:
                        num = continuous_random(l_0+i*stride,l_0+i*stride+length)
                        v = vector[num[0]:num[1], :]
                        if np.shape(v)!=(8, 2048):
                            continue;
                        window_list.append((v, [num[0],num[1]], f.split(".")[0]))

        # 주어진 window_lengths에 따라 sliding window 적용
        for length in window_lengths:
            stride = int(length * 0.25)  # stride는 window length의 25%
            num_windows = (vector.shape[0] - length) // stride + 1  # 가능한 window 개수 계산
            for i in range(num_windows):
                random_value = random.random()
                if random_value >= thresh_hold:
                    continue
                if length == 8:
                    v = vector[i*stride:i*stride+length, :]
                    if np.shape(v)!=(8, 2048):
                        continue;
                    window_list.append((v, [i*stride,i*stride+length], f.split(".")[0]))
                else:
                    num = continuous_random(i*stride,i*stride+length) 
                    v = vector[num[0]:num[1], :]
                    if np.shape(v)!=(8, 2048):
                        continue;
                    window_list.append((v, [num[0],num[1]], f.split(".")[0]))


        window_list = random.sample(window_list, num_samples)

        # 분할된 window들의 리스트 출력
        if subset=="train": # train dataset
            tresult_lst.extend(save_window(window_list, data["databases"][f.split(".")[0]])[0])
            tresultL_lst.extend(save_window(window_list, data["databases"][f.split(".")[0]])[1])

        elif subset=="val": # validation dataset
            vresult_lst.extend(save_window(window_list, data["databases"][f.split(".")[0]])[0])
            vresultL_lst.extend(save_window(window_list, data["databases"][f.split(".")[0]])[1])

        print(len(window_list))  # 총 window 개수

    # 리스트 저장
    if only=="1":
        with open("train.pkl", 'wb') as f:
            pickle.dump(tresult_lst, f)
        with open("trainL.pkl", 'wb') as f:
            pickle.dump(tresultL_lst, f)

    if only=="2":
        with open("val.pkl", 'wb') as f:
            pickle.dump(vresult_lst, f)
        with open("valL.pkl", 'wb') as f:
            pickle.dump(vresultL_lst, f)

if __name__=="__main__":
    main(only="1");