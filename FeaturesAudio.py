# import argparse
# import os
# import glob
# #from numpy import genfromtxty
# #from pydub import AudioSegment
# import numpy as np
# import torchaudio
# #import tensorflow as tf
# #import tensorflow_hub as hub
# import torch
# from FeaturesAudio_utils import video_info  # 비디오를 받아서 비디오의 정보를 프린트하고 프레임 전체 개수 및 fps를 반환
# from tqdm import tqdm
# import torch

# model = torch.hub.load('harritaylor/torchvggish', 'vggish')
# model.eval()




# def getFeature(videoname, filename, video_base_path, audio_base_path):
#     frameCnt, fps = video_info(video_base_path + videoname)
#     n_frames = 16 # 16frame 단위로 음성 피쳐를 제작한다.
#     snippet_size = int(16000 * n_frames/fps)
    
#     #frameCnt = frames[filename.split('.')[0]] 이전 코드

#     # Convert m4a to wav
#     #file = AudioSegment.from_file(input_dir + '/' + filename, "m4a")
    
#     filename = filename.split('.')[0]
#     #file.export(filename + '.wav', format="wav")

#     # Initialize Feature Vector
#     #featureVec = tf.Variable([[0 for i in range(128)]], dtype='float32')

#     # Load audio file as tensor
#     audio, sr = torchaudio.load(audio_base_path + filename + '.wav')
#     print(sr)
#     # Convert to mono
#     audio = audio.mean(axis=0)
#     #print(audio.shape)
#     # Resample to 16kHz
#     audio = torchaudio.transforms.Resample(sr, 16000)(audio.view(1, -1))[0]
#     #print(audio.shape)
#     # Iterate over all snippets and extract feature vector
#     pointerr = len(audio) // frameCnt
#     frameSize = len(audio) // frameCnt
#     #print(pointerr)
#     #print(frameSize)
#     print(" frame 별 피쳐 생산 시작 ")

#     for i in tqdm(range(frameCnt)):
#         if i % (frameCnt//100) == 0 and i!=0 and (i // (frameCnt//100)) % 10 == 0:
#             print( i // (frameCnt//100), "% 완료했습니다." )

#         # Get audio segment b/w start_time and end_time
#         # if len(audio)<=snippet_size*i:
#         #     break;
#         chunk = audio[max(0, snippet_size*i):min(len(audio), snippet_size * (i+1))]
#         # if len(chunk) < snippet_size:
#         #     break;
#         #     chunk = np.pad(chunk, pad_width=(
#         #         0, snippet_size - len(chunk)), mode='constant', constant_values=0)
#         #     # chunk = tf.convert_to_tensor(np.pad(chunk, pad_width=(
#         #     #     0, snippet_size - len(chunk)), mode='constant', constant_values=0))

#         # Extract feature vector sequence
#         print("chunk", chunk)
#         feature = model.forward(chunk)
#         #feature = vggmodel(chunk)
#         print("feature", feature)
#         #print(np.shape(feature))
#         # Combine vector sequences by taking mean to represent whole segment. (i.e. convert (Ax128) -> (1x128))
#         if len(feature.shape) == 2:
#             feature = tf.reduce_mean(feature, axis=0)

#         # Concatenate to temporal feature vector sequence
#         featureVec = tf.concat([featureVec, [feature]], 0)
#         pointerr += snippet_size

#     # Removing first row with zeroes
#     featureVec = featureVec[1:].numpy()

#     #os.remove(filename + ".wav") 보류

#     # Save as csv
#     # np.savetxt(output_dir + '/' + filename + ".csv", featureVec, delimiter=",",
#     #            header=','.join(['f' + str(i) for i in range(featureVec.shape[1])]))
#     np.save(output_dir + '/' + filename, featureVec)
#     print(filename + ' Done')

# if __name__ =="__main__":

#     parser = argparse.ArgumentParser(
#         description='Tool to extract audio feature sequence using VGGish')

#     parser.add_argument(
#         '--input',
#         default='/home/work/TAL_FineGym/Data_audio',
#         help='provide the input directory containing audio files (default: ActivityNETaudio)'
#     )

#     parser.add_argument(
#         '--video_input',
#         default='/home/work/TAL_FineGym/Data_video',
#         help='provide the input directory containing audio files (default: ActivityNETaudio)'
#     )

#     parser.add_argument(
#         '--output',
#         default='/home/work/TAL_FineGym/Feature_Audio',
#         help='path to save VGGish features (default: VGGishFeatures)'
#     )
#     # 영상마다 fps가 달라서 영상마다 다르게 처리 필요
#     # parser.add_argument(
#     #     '--snippet_size',
#     #     default=1.2,
#     #     help='snippet size in seconds (default: 1.2)'
#     # )

#     my_namespace = parser.parse_args()
#     input_dir = my_namespace.input
#     video_dir = my_namespace.video_input
#     output_dir = my_namespace.output

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # VGGish feature extractor model
#     # 이 모델의 경우 텐서플로우허브에서 제공하며 수많은 영상의 행동을 분류해주는 피쳐를 추출할 수 있도록 학습한 모델 파일
#     # 따라서 diverse하며, domain에 사용하기 위해 finefunning을 할 경우 더 좋을 것이다.
#     #vggmodel = hub.load('https://tfhub.dev/google/vggish/1')
#     # https://tfhub.dev/google/vggish/1

#     # Snippet size (In terms of no. of frames). 원래는 sample_rate가 44100임 근데 너무 커지니까 16000으로 변환해서 적용
#     # Snippet size는 피쳐의 구간을 얼마만큼으로 설정할지이다.
#     # 예를 들어 16frame 간격으로 하고 싶고 16frame이 1.2초라면 snuppet_size는 1.2가 된다.
#     # 나 같은 경우는 영상마다 fps가 다르므로 16/fps가 snippet_size

#     # snippet_size = int(16000 * my_namespace.snippet_size)

#     # data = np.genfromtxt('video_info.csv', delimiter=',', dtype=str)
#     # frames = {}
#     # for i in data[1:]:
#     #     frames[i[0]] = int(i[1])
#     # i[0]은 파일 name.mp4, i[1]은 frameCnt
#     # frameCnt는 정확히 무슨 의미? => 정말 전체 프레임 수같음.
#     # Returns feature sequence from audio 'filename'




#     # Read all files
#     fileNames = glob.glob(input_dir + "/*")
#     fileNames = [os.path.basename(file) for file in fileNames]

#     videoNames = glob.glob(video_dir + "/*")
#     videoNames = [os.path.basename(file) for file in videoNames]

#     videoNames = [x.split('.')[0]+".mp4" for x in fileNames] # audio(wav)파일이 존재하는 것에 대해서만 진행. 이에 맞춰 videofile조정
#     # video 파일은 280개 가량 존재

#     videoNames.sort() # 순서 쌍 맞춰주기
#     fileNames.sort()

#     # fileNames.sort()
#     # fileNames = fileNames[int(len(fileNames) / 2):]


#     print(str(len(fileNames)) + " files found...")


#     # Extract temporal feature sequence for all audio files.
#     video_base_path = "/home/work/TAL_FineGym/Data_video/"
#     audio_base_path = "/home/work/TAL_FineGym/Data_audio/"

#     for video,file in zip(videoNames,fileNames):
#         if video.split('.')[0] != file.split('.')[0]:
#             print(video, file)
#             print("비디오 파일과 음성 파일 명이 일치하지 않습니다.")
#             break;
#         else:
#             print(video, file)
#             getFeature(video, file, video_base_path, audio_base_path)


import torch
from torchvggish_gpu import vggish
import vggish_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For GPU support, the device must be cuda

embedding_model = vggish()
embedding_model.to(device)
embedding_model.eval()
example = vggish_input.wavfile_to_examples("/home/work/TAL_FineGym/Data_audio/_E21jlKax8M.wav")
example = example.to(device)
audio_embeddings = embedding_model.forward(example)