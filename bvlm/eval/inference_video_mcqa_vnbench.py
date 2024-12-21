import os
import re
import math
import json
import copy
import argparse
import warnings
import traceback

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]





class VNBenchDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, data_list, processor):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, line['video'])

        try:
            video_tensor = self.processor(video_path)
            num_frames = video_tensor.shape[0]
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_path}')
            video_tensor = None
            num_frames = 0

        print(video_tensor.shape, video_path)
        return {
            'video': video_tensor,
            'record': line,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    rcs = [x['record'] for x in batch]
    return vid, rcs



def build_videomme_eval(args, processor):
    # convert parquet to json
    # questions = load_parquet(args.question_file)
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VNBenchDataset(args.video_folder, questions, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


def vnbench_dump(record, instruct, options, output):
    letters = ['A', 'B', 'C', 'D']

    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(vid, instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2
    
    return letters[pred_idx]


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_videomme_eval(args, processor['video'])

    # Iterate over each sample in the ground truth file
    for i, (videos, records) in enumerate(tqdm(val_loader)):
        video_tensor  = videos[0]
        record = records[0]

        new_record = copy.deepcopy(record)

        if video_tensor is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            continue
        else:
            new_record['missing'] = False

        q = record['question']
        choices = record['options']
        letters = []
        # options = record['options']
        
        # instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
        instruct = ""
        instruct += f"{q}\n"
        for cho_idx, cho in enumerate(choices):
            letters.append(f"{chr(ord('A') + cho_idx)}")
            instruct += f"({chr(ord('A') + cho_idx)}) {cho}\n"
        # instruct += "The best option is: "
        instruct += "Answer with the option\'s letter from the given choices directly"
        output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video', do_sample=False, temperature=0.0, top_p=0.9)
        
        new_record['pred'] = vnbench_dump(record, instruct, choices, output)
        

           

        ans_file.write(json.dumps(new_record) + ",\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
