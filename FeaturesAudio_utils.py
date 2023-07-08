#import torch
#import torchaudio
#import librosa
#import numpy as np
#import moviepy.editor as mp
# import tensorflow as tf
# import tensorflow_hub as hub
import cv2

# load VGGish model
# model_path = 'path/to/vggish_model'
# model = hub.load(model_path)


def video_info(infilename):

    cap = cv2.VideoCapture(infilename)

    if not cap.isOpened():
        print("could not open :", infilename)
        exit(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('movie length : ', length, "(frame_counts)")
    print('fidelity width : ', width)
    print('fidelity height : ', height)
    print('fps : ', fps)
    #print("frames count : ", fps * length )
    return length, fps


# # AV file
# filename = "/Users/gominseong/Desktop/Video-of-a-gym-math-Temporal-Action-Localization/I3D_Feature_Extraction_resnet/audio/0jqn1vxdhls.wav"
# waveform, sample_rate = torchaudio.load(filename)
# print(len(waveform[0]), sample_rate, "**")

# # set the number of frames per feature
# num_frames = video_info(
#     "/home/work/TAL_FineGym/FineGymData/movie/0jqn1vxdhls.mp4")

# # length per audio features for setting rgb features(I3D)
# samples_per_frame = int(len(waveform[0]) / num_frames * 16)
# print(num_frames/(len(waveform[0])/sample_rate), "frame 율.")

# # len(waveform[0]) 오디오 데이터의 길이를 샘플 단위
# # sample_rate 초마다 몇개의 음성 피쳐개수가 들어가게 되는지.
# # (len(waveform[0])/sample_rate) 는 초마다의 오디오 데이터 길이

# # features = []

# # iterate over the waveform, extracting a feature for every num_frames
# # for i in range(0, len(waveform[0]), samples_per_frame):
# #     start = i
# #     end = i + samples_per_frame

# #     # check if there are enough samples left to form a full frame
# #     if end <= len(waveform[0]):
# #         # extract the samples for this frame
# #         frame = waveform[:, start:end]
# #         # extract the mel spectrogram from each frame
# #         #mel_specgrams = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(frame)
# #         #print(mel_specgrams.shape)
# #         # calculate the feature for this frame using the vaggish model
# #         #feature = # your code to calculate the feature using the vaggish model
# #         S=librosa.feature.melspectrogram(frame)
# #         log_S = librosa.power_to_db(S, ref=np.max)
# #         input_tensor = torch.Tensor(np.expand_dims(log_S, axis=0)).unsqueeze(0)
# #         print(input_tensor.shape)
# #         print("****")
# #         # Compute embeddings using VGGish model
# #         with torch.no_grad():
# #             embeddings = vggish_model(input_tensor).squeeze()
# #         feature = embeddings.cpu().numpy()
# #         # append the feature to the list of features
# #         features.append(feature)

# # convert the list of features to a PyTorch tensor
# # features_tensor = torch.Tensor(features)

# # print the shape of the features tensor
# # print(features_tensor.shape)
