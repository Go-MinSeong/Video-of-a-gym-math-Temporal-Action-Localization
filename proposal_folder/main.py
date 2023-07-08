import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from classifier_4 import ConvNet
from utils import calculate_accuracy
import pickle
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F
import sys
import random
import gc
random.seed(0)
# 기존에 출력되던 내용을 파일에 저장하기 위한 코드
#sys.stdout = open('result.txt', 'w')

# lr 0.01 conv2 학습 안됨. 로스 멈춤
# lr 0.01 conv1 학습 안됨. 로스 멈춤
# lr 0.1 conv1 학습 안됨. 로스 멈춤
# lr 0.001 conv3 정확도가 83%
# GPU 할당
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    # 길이 기준으로 내림차순 정렬
    batch.sort(key=lambda x: len(x), reverse=True)
    # 패딩
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded_batch

def main(epoch, save_txt):
    #res = open(save_txt,'a')
    # 데이터 불러오기
    with open("/home/work/TAL_FineGym/proposal_folder/train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    ct = len(train_data)
    idx=random.sample(list(range(ct)), ct//1)
    idx.sort(reverse=True)
    lst=[]
    for i in idx:
        lst.append(train_data[i])
        del train_data[i:]
    train_data = torch.tensor(lst); del lst
    print(1)
    with open("/home/work/TAL_FineGym/proposal_folder/trainL.pkl", 'rb') as f:
        train_labels = pickle.load(f)
    lst=[]
    for i in idx:
        lst.append(train_labels[i])
        del train_labels[i:]
    train_labels = torch.tensor(lst); del lst; del idx
    # 135660
    #res.write(str((len(train_data)))) 메모리 관리
    #res.write(str((len(train_labels))))
    print("train 준비 완료");
    #res.close()

    with open("/home/work/TAL_FineGym/proposal_folder/val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    ct = len(val_data)
    idx=random.sample(list(range(ct)), ct//1)
    idx.sort(reverse=True)
    lst=[]
    for i in idx:
        lst.append(val_data[i])
        del val_data[i:]
    val_data = torch.tensor(lst); del lst

    val_data = torch.tensor(val_data)
    with open("/home/work/TAL_FineGym/proposal_folder/valL.pkl", 'rb') as f:
        val_labels = pickle.load(f)
    lst=[]
    for i in idx:
        lst.append(val_labels[i])
        del val_labels[i:]
    val_labels = torch.tensor(lst); del lst; del idx

    res = open(save_txt,'a')
    res.write("train 준비 완료")
    print("val 준비 완료"); res.write("val 준비 완료")
    print("TRAIN_Dataset 개수 : ",len(train_data)) ; res.write(f"TRAIN_Dataset 개수 : {len(train_data)}")
    print("VAL_Dataset 개수 : ",len(val_data)); res.write(f"VAL_Dataset 개수 : {len(val_data)}")

    # DataLoader를 이용하여 학습 데이터셋 구성
    batch_size = 4

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 모델 학습
    model = ConvNet()  # 학습할 모델 생성
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()  # 손실 함수
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    print(learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=300000, gamma=0.1)

    epochs = 50  # 학습할 에폭 수
    min_loss=9999
    # 모델 객체와 모델의 가중치를 저장할 파일 경로 정의
    PATH = "model3.pt"

    print("학습시작") ; res.write("학습시작")
    res.close()
    for epoch in range(epochs):
        res = open(save_txt,'a')
        running_loss = 0.0
        dr=0
        for i, (inputs, labels) in enumerate(train_loader):
            if i == len(train_loader)-1:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # gradient 초기화
            outputs = model(inputs)  # 모델 예측
            loss = loss_fn(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()  # 손실 누적
            print(f'{i} / {len(train_loader)} 진행 완료, 현재 로스 : {loss.item()}',  end="\r")
            if i % (len(train_loader)//3)==1:
                dr+=1
                print("3분의",dr-1," 진행되었으며 지금까지의 loss는 ",running_loss/(i+1),"입니다.",  end="\r")
                res.write(f'3분의 {dr-1}  진행되었으며 지금까지의 loss는 {running_loss/(i+1)} 입니다.' +"\n")
        #print(outputs[:5], labels[:5])
        print('[Epoch %d] 완. train loss: %.3f' % (epoch+1, running_loss/len(train_loader)))
        with torch.no_grad():
            model.eval()
            test_loss = 0 
            correct = 0
            for i, (inputs, labels) in enumerate(val_loader):
                if i == len(val_loader)-1:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()  # 손실 누적
                accuracy = calculate_accuracy(outputs, labels)
                correct += accuracy
            print('[Epoch %d] 완. val loss: %.3f' % (epoch+1, test_loss/len(val_loader)))
            res.write('[Epoch %d] 완. val loss: %.3f' % (epoch+1, test_loss/len(val_loader)) +'\n')
            print(f"The accuracy is {correct/len(val_loader) :.2f}%")
            res.write(f"The accuracy is {correct/len(val_loader) :.2f}%" +'\n')
            res.close()

        if min_loss > test_loss/len(val_loader):
            min_loss = test_loss/len(val_loader)
            torch.save(model,PATH)
            print(epoch+1, "모델 저장")
    # 파일 닫기
    #sys.stdout.close()

if __name__=="__main__":
    main(epoch=20, save_txt="/home/work/TAL_FineGym/proposal_folder/proposal_process.txt")