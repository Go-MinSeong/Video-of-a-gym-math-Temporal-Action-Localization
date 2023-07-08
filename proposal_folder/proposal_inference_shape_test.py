import numpy as np 
import pandas as pd
from classifier_3 import ConvNet
import torch
import json
import os
from tqdm import tqdm
import random
from utils import right_proposal, cal_iou, save_window, compute_iou, save_window_v4, video_info, nms_v2


def continuous_random(x, y):
    num=random.randrange(x,y-7)
    return num,num+8
def features_to_proposal(f):
    # sliding window를 거치면서 분할된 window들을 저장할 리스트
    window_list = []
    vector = np.load(dir_home+"/"+f)
    print(vector.shape)
    window_lengths = [8, 12, 16, 24, 36, 48, 64, 96, 128]  # sliding window의 길이

    # 주어진 window_lengths에 따라 sliding window 적용
    for length in window_lengths:
        stride = int(length * 0.25)  # stride는 window length의 25%
        num_windows = (vector.shape[0] - length) // stride + 1  # 가능한 window 개수 계산
        for i in range(num_windows):
            if length ==8:
                # if random.random() < 0.9:
                #     continue
                v = vector[i*stride:i*stride+length, :]
                if np.shape(v)!=(8, 2048):
                    continue;
                window_list.append((v, [i*stride,i*stride+length]))
            else:
                # if random.random() < 0.3:
                #     continue
                num = continuous_random(i*stride,i*stride+length) 
                v = vector[num[0]:num[1], :]
                if np.shape(v)!=(8, 2048):
                    continue;
                window_list.append((v, [num[0],num[1]]))
    print("추출된 proposal의 갯수 : ", len(window_list))
    return window_list

def To_model(model, window_list, data, threshhold=0.1):
    result=[]
    model.eval()
    for data in window_list:
        dt = torch.tensor(data[0]).unsqueeze(0)
        with torch.no_grad():
            output = model(dt.to(device))
        #print(output)
        if output[0][0]>=threshhold:
            result.append([output[0][0], data[1]])
    print("proposal nms전 개수 : ",len(result))
    return result

def writing(fl, lst, data, name):
    with open(fl, 'r') as file:
        content = file.read()
        count = content.count('#')
    fps = video_info(name)["fps"]
    res = open(fl,'a')
    res.write("#%d\n" % count)
    res.write(str(name)+"\n")
    res.write(str(int(data["databases"][name]["duration"]*fps))+"\n")
    res.write(str(1)+"\n")
    ls = data["databases"][name]["annotation"]
    res.write(str(len(ls))+"\n")
    for r in ls:
        res.write(str(r["label"][0]))
        res.write(" ")
        res.write(str(int(r["segment"][0]*fps)))
        res.write(" ")
        res.write(str(int(r["segment"][1]*fps))+"\n")
    st = save_window_v4(lst, data["databases"][name], name)
    # [iou, iou2, 후보 구간, 클래스] for x in lst
    res.write(str(len(st))+"\n")
    for s in st:
        res.write(str(s[3][0]))
        res.write(" ")
        res.write(str(np.round(s[1], 4)))
        res.write(" ")
        res.write(str(np.round(s[0], 4)))
        res.write(" ")
        res.write(str(int(s[2][0])))
        res.write(" ")
        res.write(str(int(s[2][1])))
        res.write("\n")
    res.close()

def main(model, f, data, dir_home):
    global final_result;
    # json 파일 불러오기
    # train, val 구분
    subset = data["databases"][f.split(".")[0]]["subset"]
    if subset == "train":
        return
        fl = "train_prop.txt"
    else:
        #continue; # test 파일도 작성하기 위해서는 이부분을 제거 바랍니다.
        fl = "test_prop2.txt"
    with open(fl, 'r') as file:
        content = file.read()
        count = content.count('#')

    random.seed(0)
    # model, window_list
    proposal_list = features_to_proposal(f)
    #print("proposal generation 완료")
    # features_to_proposal는 해당 넘파이 피쳐를 넣게 되면 후보 proposal전체 리스트를 반환함.
    #score_list = To_model(model, proposal_list, data)
    #print("proposal model 출력 완료")
    # To_model은 후보 proposal이 학습되어진 모델에 들어가서, iou를 예측하여옴.
    # 출력은 (iou, [start, end]) for x in lst)
    #print(score_list[0])
    #lst = nms_v4(proposal_list, iou_thresh=0.6, top_k=100)
    #print("proposal nms 완료")
    # nms는 proposal끼리 유사한 것을 제거하고 최대 200개의 proposal로 간추림.
    # 출력은 최대 200개의 (iou, [start, end]) for x in lst)

    writing(fl, proposal_list, data, f.split(".")[0])
    print("proposal writing 완료")


if __name__=="__main__":
    final_result=[]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dir_home = "/home/work/TAL_FineGym/Feature_RGB"
    # 모델 지정
    model = ConvNet()  # 학습할 모델 생성
    model=torch.load('model.pt') # 모델 불러오기
    model.to(device)
    f_lst =  os.listdir(dir_home)[:]
    #for f in f_lst:
    #    final_result.append(main(model, f)); # model, 영상의 rgb 피쳐.npy
    with open('/home/work/TAL_FineGym/Label/finegym_annotation.json', 'r') as k:
        data = json.load(k)
    for fe in tqdm(f_lst):
        print(fe, "start")
        main(model, fe, data, dir_home)

