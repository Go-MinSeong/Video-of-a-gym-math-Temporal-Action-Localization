import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable


def load_frame(frame_file):
	data = Image.open(frame_file) # 이미지 파일을 열고 리사이즈 진행.
	#data = data.resize((340, 256), Image.ANTIALIAS)
	data = np.array(data)
	data = data.astype(float)
	data = (data * 2 / 255) - 1
	assert(data.max()<=1.0)   # 데이터를 -1 ~ 1 로 전환
	assert(data.min()>=-1.0)
	return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (224,224,3))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
	return batch_data


def oversample_data(data):
	data_flip = np.array(data[:,:,:,::-1,:])

	data_1 = np.array(data[:, :, :224, :224, :])
	data_2 = np.array(data[:, :, :224, -224:, :])
	data_3 = np.array(data[:, :, 16:240, 58:282, :])
	data_4 = np.array(data[:, :, -224:, :224, :])
	data_5 = np.array(data[:, :, -224:, -224:, :])

	data_f_1 = np.array(data_flip[:, :, :224, :224, :])
	data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
	data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
	data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
	data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

	return [data_1, data_2, data_3, data_4, data_5,
		data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def run(i3d, frequency, frames_dir, batch_size, sample_mode):
	assert(sample_mode in ['oversample', 'center_crop'])
	print("batchsize", batch_size) # 배치 사이즈 출력
	chunk_size = 16 # 덩어리 사이즈  피쳐의 크기 채널딴.
	def forward_batch(b_data):
		b_data = b_data.transpose([0, 4, 1, 2, 3])
		b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 20x3x16x224x224
		with torch.no_grad():  # 테스트
			b_data = Variable(b_data.cuda()).float()
			inp = {'frames': b_data}
			features = i3d(inp)
		return features.cpu().numpy()

	rgb_files = natsorted([i for i in os.listdir(frames_dir)])  # 텍스트로 된 숫자 파일명 정리
	#print(rgb_files[:100])
	frame_cnt = len(rgb_files) # 프레임 몇개인지
	print("frame 개수 : " , frame_cnt)
	# Cut frames
	assert(frame_cnt > chunk_size)  # 덩어리보다 프레임이 많아야 한다.
	clipped_length = frame_cnt - chunk_size # 덩어리 개수
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk  정확한 덩어리 개수??
	frame_indices = [] # Frames to chunks

	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
	# 리스트에 덩어리 별로 모음 
	#수정 후
	# for i in range(clipped_length):
	# 	frame_indices.append([j for j in range(i, i + chunk_size)])

	frame_indices = np.array(frame_indices)
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]
	else:
		full_features = [[]]


	for batch_id in range(batch_num): 
		print(batch_num,"분의 ",batch_id," 완료", end="\r")
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
		if(sample_mode == 'oversample'):
		   batch_data_ten_crop = oversample_data(batch_data)
		   for i in range(10):
			   assert(batch_data_ten_crop[i].shape[-2]==224)
			   assert(batch_data_ten_crop[i].shape[-3]==224)
			   temp = forward_batch(batch_data_ten_crop[i])
			   full_features[i].append(temp)

		elif(sample_mode == 'center_crop'):
			#batch_data = batch_data[:,:,16:240,58:282,:]
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			temp = forward_batch(batch_data)
			full_features[0].append(temp)
	
	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	full_features = full_features[:,:,:,0,0,0]
	full_features = np.array(full_features).transpose([1,0,2])
	return full_features
