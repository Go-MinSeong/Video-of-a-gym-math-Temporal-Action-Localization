import subprocess
import os
import shutil
from tqdm import tqdm
from pathlib import Path
import zipfile

def zip_extract():
    video_lst = os.listdir(
        "/home/work/golf/Mymodel/I3D_Feature_Extraction_resnet/output/output_2")
    video_lst_2 = os.listdir(
        "/home/work/golf/Mymodel/I3D_Feature_Extraction_resnet/output/output_3") 
    video_lst = list(set(video_lst)-set(video_lst_2))

    print(video_lst)

    for video in tqdm(video_lst[:]):
        if video == ".DS_Store":
            continue
        print(video, "압축 해체를 시작하도록 하겠습니다.")
        path = "/home/work/golf/Mymodel/I3D_Feature_Extraction_resnet/output/"
        in_path = path + "output_2/" + video + "/" + video + "_out_part_file.zip"
        out_path = path + "output_1/" + video + "_join_file.zip"
        out_path_2 = path + "output_3/" + video
        
        subprocess.call("zip -FF "+ in_path + " --out " + out_path,
                        shell=True)  # 분할 압축 파일을 하나의 압출 파일로 변환하는 과정  output1에 저장
        
        print(" 압축 완료했습니다. ")
        subprocess.call( "unzip "+out_path+" -d "+out_path_2,
            shell=True) # 하나로 만든 압축 파일을 해제하는 코드

if __name__=="__main__":
    zip_extract()
