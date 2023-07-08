# pip install pytube.  모듈 설치 필요
import requests
from tqdm import tqdm
import json
import collections
from pytube import YouTube
import os
import subprocess

# 지금 코드에서 고려해야 할 부분 다운된 영상의 해상도 일정도, frame 일정도, 음성 데이터를 처음부터 따로 할 것인지.



# video data download

def download_data(json_dir, out_dir, out_dir2, label_dir):
    no_download=[]
    #dict_title ={}
    with open(json_dir, 'r') as fp: # json file을 연다
        meta = json.load(fp, object_pairs_hook=collections.OrderedDict) # json파일을 딕셔너리 순서로 불러옴. 경기마다 리스트
    data_obj_list=list(meta.keys())

    print(len(data_obj_list), "files downloading start")

    for obj_url in tqdm(data_obj_list):

        url = 'https://www.youtube.com/watch?v='+obj_url
        yt = YouTube(url)
        try:
            fc = yt.streams.filter(file_extension='mp4', res="360p",  only_audio=False).first()
            fc.download(out_dir, filename = obj_url+".mp4")
            #dict_title[fc.default_filename] = obj_url

            # 필터링된 스트림 중 첫 번째 오디오 스트림 가져오기
            #audio_stream = yt.streams.filter(only_audio=True).first()
            # 오디오 다운로드
            #audio_stream.download(out_dir2, filename=obj_url+".wav")
            print(fc.default_filename)

        except:
            no_download.append(obj_url)
    
    #with open(label_dir, 'w') as f:
        #json.dump(dict_title, f)
        
    print(len(data_obj_list)-len(no_download), "files downloading complete")


# label file download
def main(json_dir, vidoe_dir, auido_dir, label_dir):
    url_lst = []
    url_lst.extend(['https://sdolivia.github.io/FineGym/resources/dataset/finegym_annotation_info_v1.1.json',
               'https://sdolivia.github.io/FineGym/resources/dataset/gym99_train_element_v1.1.txt',
               'https://sdolivia.github.io/FineGym/resources/dataset/gym99_val_element.txt'])
    name_lst = [name.split('/')[-1] for name in url_lst]
    for url, filename in zip(url_lst, name_lst):
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            # find the source of the video in the HTML
            # this part may vary depending on the website and the structure of the HTML
            # but a common approach is to search for the src attribute in a video tag, such as <video src="video_url">
            if response.status_code == 200:
                with open(label_dir+'/'+filename, "wb") as f:
                    f.write(response.content)
            else:
                print("Failed to retrieve video contents")
        else:
            print("Failed to retrieve HTML contents")
    download_data(json_dir, vidoe_dir, auido_dir, label_dir)


if __name__ == "__main__":
  main( 
'/home/work/TAL_FineGym/Label/finegym_annotation_info_v1.1.json',
 '/home/work/TAL_FineGym/Data_video',
 '/home/work/TAL_FineGym/Data_audio',
 '/home/work/TAL_FineGym/Label')
