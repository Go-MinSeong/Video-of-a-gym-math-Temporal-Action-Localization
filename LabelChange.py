# pip install pytube.  모듈 설치 필요
from tqdm import tqdm
import json
import os
import pickle
from file_name import video_duration

# label file download
def main(label_dir, source_folder, used_data, save_dir):
    # 새로운 딕셔너리 생성
    new_data={}
    new_data["classes"] = ['0','1','2','3','4','5','6','7','8','9','10']
    fine_new_data={}

    # 사용할 영상 정의
    vd_ls = os.listdir(source_folder)
    #vd_ls = os.listdir(used_data)
    video_file_list = [ x.split(".")[0] for x in vd_ls]
    print(video_file_list)
    
    # JSON 파일 열기
    with open(label_dir, 'r') as f:
        data = json.load(f)
    val_num = int(len(video_file_list)*0.8)
    for i, video in enumerate(video_file_list):
        dict_1 = {}
        # train_test split
        if i>val_num:
            split = "val"
        else:
            split = "train"

        vf = source_folder + "/" + video+".mp4"
        duration = video_duration(vf)
        dict_1["duration"] = duration
        dict_1["subset"] = split

        # timestamp setting
        fine_list_1 = []
        fine_list = list(data[video].values())
        #print("*****f",fine_list)
        for p, action in enumerate(fine_list):
            #print("******")
            #print(fine_list)
            fine_dict  = {}
            fine_dict["segment"] = action["timestamps"][0]
            #print(fine_dict["segment"])
            fine_dict["label"] = [str(x) for x in [action["event"]]]
            fine_dict["num_shots"] = 1
            fine_list_1.append(fine_dict)
            #print(action)
            #print(fine_dict)
            #if p==1:
            #    break;
        #if p==1:
        #    print(fine_list_1)
        #    break;
        #print("dededed", fine_list_1)
        dict_1["annotation"] = fine_list_1
        fine_new_data[video] = dict_1
    new_data["databases"] = fine_new_data

    # 딕셔너리를 JSON 문자열로 변환
    json_string = json.dumps(new_data)

    # JSON 문자열을 파일로 저장
    with open(save_dir+"/finegym_annotation.json", "w") as f:
        f.write(json_string)





if __name__ == "__main__":
    main("/home/work/TAL_FineGym/Label/finegym_annotation_info_v1.1.json",
    '/home/work/TAL_FineGym/Data_video',
    "/home/work/TAL_FineGym/Feature_RGB",
    "/home/work/TAL_FineGym/Label")


    # "-HvXd_vqJ5A": { "E_001781_001789": { "event": 1, "segments": null, 
    # "timestamps": [[1781.76, 1789.16]]},
    #     "E_001805_001812": {
    #         "event": 1,
    #         "segments": null,
    #         "timestamps": [
    #             [
    #                 1805.48,
    #                 1812.92
    #             ]
    #         ]
    #     }}