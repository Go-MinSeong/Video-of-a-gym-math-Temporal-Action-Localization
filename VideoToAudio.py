import subprocess
import os
import sys


def convert_video_to_audio_ffmpeg(video_file, out, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    # print(filename, ext)
    # out = out + "/" + filename.split("/")[-1]
    # print(out)
    #command = "ffmpeg -i {} -ar 44100 -vn {}".format(video_file, os.path.join(out+"/"+filename.split("/")[-1]+"."+output_ext))
    command = "ffmpeg -i {} -ar 44100 -vn {}".format(video_file, filename.split("/")[-1]+"."+output_ext)
    cmd = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", out+"/"+filename.split("/")[-1]+"."+output_ext]
    subprocess.run(cmd,check=True)

if __name__ == "__main__":
    vf = "/home/work/TAL_FineGym/Data_video"
    out = "/home/work/TAL_FineGym/Data_audio"
    lst = os.listdir(vf)
    lst1 = os.listdir(out)
    result = 0
    result_1 = []
    print(len(out))
    for v in lst1:
        # print(v)
        if v.split(".")[1] == "wav":
            result_1.append(v.split(".")[0])
    for v in lst:
        if v.split(".")[0] in result_1:
            continue
        if v.split(".")[1] == "mp4" or v.split(".")[1] == "mkv":
            result += 1
            print(v)
            print(result)
            convert_video_to_audio_ffmpeg(vf + "/" + v, out)
