import os
import glob
import math
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image, TarIO
from tqdm import tqdm
from videollama2.model.eva_vit import EVAVisionTowerLavis
import tarfile
import io
import re
import datetime
from decord import VideoReader, cpu
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir", required=True, help="Path to read the videos from.")
    parser.add_argument("--feat_dir", required=True, help="The output dir to save the features in.")
    parser.add_argument("--fps", type=int, required=True, help="The output dir to save the features in.")
    parser.add_argument("--vision_tower", type=str,default="./model_zoo/LAVIS/eva_vit_g.pth", help="Vision backbone to process the video.")
    parser.add_argument("--image_processor", type=str, default="openai/clip-vit-large-patch14-336", help="Image processor to pre-process the video.")
    parser.add_argument("--chunk_idx", type=int, default=0, help="index of chunk.")
    parser.add_argument("--num_chunks", type=int, default=1, help="number of chunk.")
    parser.add_argument("--infer_batch", required=False, type=int, default=48,
                        help="Number of frames/images to perform batch inference.")
    args = parser.parse_args()
    return args


def get_second(time_str):
    time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S,%f')
    seconds = (time_obj - datetime.datetime(1900, 1, 1)).total_seconds()
    return seconds


def load_subtitles(file_path):

    def check(subtitle):
        if subtitle.get('text', None) is None: return False 
        if subtitle.get('start_time', None) is None: return False 
        if subtitle.get('end_time', None) is None: return False 
        return True

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    subtitles = []
    subtitle = {}
    for line in lines:
        line = line.replace('\x00', '').strip()
        line = line.replace('...', '').strip()
        if not line: continue
        if line.isdigit():
            if check(subtitle):
                subtitles.append(subtitle)
                subtitle = {}
        elif ' --> ' in line:
            start, end = line.split(' --> ')
            subtitle['start_time'] = get_second(start)
            subtitle['end_time'] = get_second(end)
        else:
            if subtitle.get('text', None):
                subtitle['text'] += ' ' + line
            else:
                subtitle['text'] = line
    
    if check(subtitle):
        subtitles.append(subtitle)
        subtitle = {}
    
    return subtitles


def load_video(video_path):
    video_file = tarfile.open(video_path, 'r')
    image_file = [x for x in video_file.getmembers() if '.jpg' in x.name]
    image_data = [video_file.extractfile(x).read() for x in image_file]
    image_data = [Image.open(io.BytesIO(x)) for x in image_data]

    file_name = [x.name for x in image_file]
    indexed_list = list(enumerate(file_name))
    indexed_list = sorted(indexed_list, key=lambda x: x[1])
    image_data = [image_data[x] for x, _ in indexed_list]
    return image_data


def load_video_input(subtitles, frame_time):
    point = 0 
    text_list = [[] for i in range(len(frame_time))]
    for item in subtitles:
        end_time = item['end_time']
        text = item['text']
        while point < len(frame_time) and frame_time[point] < end_time:
            point += 1
        text_list[point - 1].append(text)
    video_input = ''
    for text in text_list:
        video_input += '<image>' + ' '.join(text)
    return video_input


def main():
    args = parse_args()
    video_dir = args.video_dir
    feat_dir = args.feat_dir
    infer_batch = args.infer_batch
    os.makedirs(feat_dir, exist_ok=True)

    # Initialize the CLIP model
    vision_tower = EVAVisionTowerLavis(args.vision_tower, args.image_processor, args=None).cuda()
    vision_tower.eval()
    image_processor = vision_tower.image_processor

    video_files = os.listdir(args.video_dir)
    video_files = get_chunk(video_files, args.num_chunks, args.chunk_idx)


    for video_id in tqdm(video_files):

        if "mp4" not in video_id:
            continue


        feat_path = os.path.join(feat_dir, f"{video_id}.npy")
        if os.path.exists(feat_path): continue

        # raise

        video_file = os.path.join(args.video_dir, video_id)
        # extract features
        vr = VideoReader(video_file, ctx=cpu(0))
        fps = args.fps
        if args.fps == -1:
            fps = vr.get_avg_fps()
        sample_fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        image_data = vr.get_batch(frame_idx).asnumpy()

        video_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half()

        n_chunk = len(video_tensor)
        video_features = torch.FloatTensor(n_chunk, 577, 1408).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(infer_batch)))
        for i in range(n_iter):
            min_ind = i * infer_batch
            max_ind = min((i + 1) * infer_batch, n_chunk)
            video_batch = video_tensor[min_ind:max_ind].cuda()
            batch_features = vision_tower(video_batch)
            video_features[min_ind:max_ind] = batch_features.detach().cpu()
        video_features = video_features.numpy().astype("float16")
        print(f"ori len: {len(vr)}, save as :{video_features.shape}")
        # video_info = dict(feats=video_features, inputs=video_input)
        np.save(feat_path, video_features)
        
        # with open(feat_path, 'wb') as f:
        #     pickle.dump(video_info, f)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
