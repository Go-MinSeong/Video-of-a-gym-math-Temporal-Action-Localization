import subprocess
import os
import shutil
from tqdm import tqdm
from pathlib import Path

video_lst = os.listdir(
    "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/π’πΎππ½ππ·/I3D_Feature_Extraction_resnet/output_3/temp")
path = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/π’πΎππ½ππ·/I3D_Feature_Extraction_resnet/output_3/temp/"
for video in tqdm(video_lst[:]):
    if video == ".DS_Store":
        continue
    print(video)
    video_path = path+video
    zip_video_folder = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/π’πΎππ½ππ·/I3D_Feature_Extraction_resnet/output/output_1/"
    zip_video_folder_2 = "/Users/gominseong/Library/CloudStorage/OneDrive-KookminUNIV/π’πΎππ½ππ·/I3D_Feature_Extraction_resnet/output/output_2/" + video + "/"

    subprocess.call("zip -r " + zip_video_folder + video + "_file.zip "+video_path,
                    shell=True)  # μΌλ¨ μ¬μ§ μ μ²΄λ₯Ό μμΆν΄μ λ§λλ μ½λ
    # zip -r out_file.zip ~/source_dir
    # κ²½λ‘μ ν΄λΉνλ ν΄λ μμ μ μμ±.
    Path(zip_video_folder_2).mkdir(parents=True, exist_ok=True)

    subprocess.call("zip -s 100m " + zip_video_folder + video + "_file.zip --out " + zip_video_folder_2 + video + "_out_part_file.zip",
                    shell=True)  # μμΆ νμΌμ 100mb μ© λλ μ λΆν  μμΆμΌλ‘ λ³ννλ μ½λ
    # zip -s 100m out_file.zip --out out_part_file.zip
    print(" μΆμΆ μμ μλ£νμ΅λλ€. ")

    os.remove(zip_video_folder + video + "_file.zip")
    shutil.rmtree(video_path)  # μΆμΆν μ΄λ―Έμ§ μ­μ 

    print(video + "  process  Done   Congratulations ~")
'''zip -s 100m out_file.zip --out out_part_file.zip
shutil.rmtree(temppath)
zip -F out_part_file.zip --out join_file.zip
unzip join_file.zip'''
