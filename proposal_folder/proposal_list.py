# 23개가 74만개
import numpy as np 
import pandas as pd
from classifier import ConvNet
import torch
import json
import os
from tqdm import tqdm
import random
from utils import right_proposal, cal_iou, save_window, save_window_v2, nms, video_info
import pickle
# train :  213 , validation :  55 으로 스플릿 진행합니다.

''' 아래와 같은 텍스트 형식을 만들어야 한다.
#0
2536
3653
1
6
15 0 165
6 159 382
6 759 890
3 1501 2320
6 2345 2740
6 1509 2059
100
3 0.552810 0.614002 1626 2757
3 0.427926 0.605519 1834 2637
'''
def main(train_true=1):
    global window_list
    result_lst=[]
    resultL_lst=[]
    features = os.listdir("/home/work/TAL_FineGym/Feature_RGB")
    dir_home = "/home/work/TAL_FineGym/Feature_RGB"

    random.seed(0)

    def continuous_random(x, y):
        num=random.randrange(x,y-3)
        return num,num+4

    # json 파일 불러오기
    with open('/home/work/TAL_FineGym/Label/finegym_annotation.json', 'r') as f:
        data = json.load(f)

    train_rate = int(len(features)*0.8)-1
    print("train : ",train_rate+1,", validation : " ,len(features)-train_rate-1, "으로 스플릿 진행합니다.")
    cot=-1
    for k, f in tqdm(enumerate(features)):
        window_list=[]
        fps = video_info(f.split(".")[0])["fps"]
        vector = np.load(dir_home+"/"+f)
        lst = data["databases"][f.split(".")[0]]["annotation"]
        lst_1 = [[u*fps//16 for u in x["segment"]] for x in lst]
        subset = data["databases"][f.split(".")[0]]["subset"]
        print(f)
        # 기록
        if subset =="train":
            if train_true==1:
                res = open("proposal_train.txt",'a')
                cot+=1
            else:
                continue;
        else:
            if train_true!=1:
                res = open("proposal_test.txt",'a')
                cot+=1
            else:
                continue;
        res.write("#%d\n" % cot)
        res.write(str(f.split(".")[0])+"\n")
        res.write(str(int(np.shape(vector)[0]*16/2.5))+"\n")
        res.write(str(1)+"\n")
        res.write(str(len(lst))+"\n")
        for r in lst:
            res.write(str(r["label"][0]))
            res.write(" ")
            res.write(str(int(r["segment"][0]*fps/2.5)))
            res.write(" ")
            res.write(str(int(r["segment"][1]*fps/2.5))+"\n")
        
        window_lengths = [4, 8, 16, 32, 64, 128, 256, 512]  # sliding window의 길이
        # 정답 어노테이션에서 추출하는 프로포설
        for length in window_lengths:
            stride = int(length * 0.25)  # stride는 window length의 25%
            for l in lst_1: # l [12, 23]
                l_0, l_1=int(l[0]), int(l[1])
                if l_1-l_0 < length:
                    continue;
                num_windows = (l_1-l_0 - length) // stride + 1

                for i in range(num_windows):
                    if l_0+i*stride+length >= vector.shape[0]:
                        continue;
                    if length == 4:
                        window_list.append(([l_0+i*stride,l_0+i*stride+length], f.split(".")[0]))
                    else:
                        num = continuous_random(l_0+i*stride,l_0+i*stride+length)
                        window_list.append(([num[0],num[1]], f.split(".")[0]))

        # 주어진 window_lengths에 따라 sliding window 적용
        for length in window_lengths:
            stride = int(length * 0.25)  # stride는 window length의 25%
            num_windows = (vector.shape[0] - length) // stride + 1  # 가능한 window 개수 계산
            for i in range(num_windows):
                if i*stride+length >= vector.shape[0]:
                    continue;
                if length ==4:
                    #print(v.shape)
                    #v = np.mean(v, axis=0, keepdims=True)
                    window_list.append(([i*stride,i*stride+length], f.split(".")[0]))
                else:
                    num = continuous_random(i*stride,i*stride+length) 
                    window_list.append(([num[0],num[1]], f.split(".")[0]))
        # 분할된 window들의 리스트 출력
        print(len(window_list))  # 총 window 개수

        result_lst.extend(save_window_v2(window_list, data["databases"][f.split(".")[0]]) )
        # iou, iou2, 후보구간, 클래스, 영상 이름

        final = nms(result_lst, top_k=200)
        res.write(str(len(final))+"\n")
        for f in final:
            res.write(str(f[3][0]))
            res.write(" ")
            res.write(str(np.round(f[0], 4)))
            res.write(" ")
            res.write(str(np.round(f[1], 4)))
            res.write(" ")
            res.write(str(int(f[2][0]*16/2.5)))
            res.write(" ")
            res.write(str(int(f[2][1]*16/2.5)))
            res.write("\n")
        res.close()

if __name__=="__main__":
    main(train_true=1);