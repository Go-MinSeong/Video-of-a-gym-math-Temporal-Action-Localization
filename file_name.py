import os

#path_dir = '/home/work/golf/FineGymData/data_folder'
#file_list = os.listdir(path_dir)

def name_changer(file_list, dir):
    for file in file_list:
        os.rename(dir+file, dir+file+".mp4")


import cv2
def video_info(infilename):
    cap = cv2.VideoCapture(infilename)
    
    if not cap.isOpened():
        infilename = infilename.split(".")[0]+".mkv"
        cap = cv2.VideoCapture(infilename)

    if not cap.isOpened():
        print("could not open :", infilename)
        exit(0)
        
    print(infilename)
 
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    print('length : ', length)
    print('width : ', width)
    print('height : ', height)
    print('fps : ', fps)


def video_duration(infilename):
    
    cap = cv2.VideoCapture(infilename)
 
    if not cap.isOpened():
        infilename = infilename.split(".")[0]+".mkv"
        cap = cv2.VideoCapture(infilename)

    if not cap.isOpened():
        print("could not open :", infilename)
        exit(0)

    print(infilename)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = length / fps

    print('dutation : ', duration)
    print('length : ', length)
    print('width : ', width)
    print('height : ', height)
    print('fps : ', fps)

    return duration

#video_info("/home/work/golf/FineGymData/movie/Wu1gxxpbF2o.mp4")
# 해상도는 640*360, fps는 25~30가량인거 같다.