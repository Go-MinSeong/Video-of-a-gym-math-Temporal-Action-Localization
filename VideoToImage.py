import cv2
import os
from tqdm import tqdm
def main():
    # 변환할 해상도 설정
    width, height = 224, 224

    # 파일이 있는 폴더 경로
    dir_path = '/home/work/TAL_FineGym/Data_video'
    out_path = '/home/work/TAL_FineGym/Data_img1'
    # 폴더 내 파일 목록 조회

    video_lst = os.listdir(dir_path)
    img_lst = set(os.listdir(out_path))
    be_lst = set([x.split('.')[0] for x in video_lst])
    lst = list(be_lst - img_lst)
    vd_lst = [x for x in video_lst if x.split('.')[0] in lst]
    print(vd_lst)

    for file_name in tqdm(vd_lst):
        print(file_name)
        file_path = os.path.join(dir_path, file_name)
        if file_name.endswith('.mp4') or file_name.endswith('.mkv'):
            # 비디오 캡쳐 객체 생성
            cap = cv2.VideoCapture(file_path)
            # 저장할 이미지 파일명 설정
            img_file_name = os.path.splitext(file_name)[0] + '_%04d.jpg'
            img_dir_path = os.path.join(out_path, os.path.splitext(file_name)[0])
            if not os.path.exists(img_dir_path):
                os.makedirs(img_dir_path)

            # 비디오 프레임을 가져와서 이미지로 저장
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (width, height))
                img_path = os.path.join(img_dir_path, img_file_name % count)
                cv2.imwrite(img_path, frame_resized)
                count += 1

            cap.release()

if __name__=="__main__":
    main()