import subprocess
import os
import shutil
from tqdm import tqdm
from pathlib import Path

video_lst = os.listdir(
    "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/output_3/temp")
path = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/output_3/temp/"
for video in tqdm(video_lst[:]):
    if video == ".DS_Store":
        continue
    print(video)
    video_path = path+video
    zip_video_folder = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/output/output_1/"
    zip_video_folder_2 = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/𝒢𝒾𝓉𝒽𝓊𝒷/I3D_Feature_Extraction_resnet/output/output_2/" + video + "/"

    subprocess.call("zip -r " + zip_video_folder + video + "_file.zip "+video_path,
                    shell=True)  # 일단 사진 전체를 압축해서 만드는 코드
    # zip -r out_file.zip ~/source_dir
    # 경로에 해당하는 폴더 없을 시 생성.
    Path(zip_video_folder_2).mkdir(parents=True, exist_ok=True)

    subprocess.call("zip -s 100m " + zip_video_folder + video + "_file.zip --out " + zip_video_folder_2 + video + "_out_part_file.zip",
                    shell=True)  # 압축 파일을 100mb 씩 나눠서 분할 압축으로 변환하는 코드
    # zip -s 100m out_file.zip --out out_part_file.zip
    print(" 추출 작업 완료했습니다. ")

    os.remove(zip_video_folder + video + "_file.zip")
    shutil.rmtree(video_path)  # 추출한 이미지 삭제

    print(video + "  process  Done   Congratulations ~")
'''zip -s 100m out_file.zip --out out_part_file.zip
shutil.rmtree(temppath)
zip -F out_part_file.zip --out join_file.zip
unzip join_file.zip'''
