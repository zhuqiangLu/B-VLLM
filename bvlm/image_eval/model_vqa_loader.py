import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llamavid.conversation import conv_templates, SeparatorStyle
# from llamavid.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import sys

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, image_processor, ):
        self.questions = questions
        self.image_folder = image_folder
        # self.tokenizer = tokenizer
        self.image_processor = image_processor

        qs = list()
        for q in questions:
            if 'video' in q['image']:
                continue 
            else:
                qs.append(q)

        self.questions = qs



    def __getitem__(self, index):
        line = self.questions[index]
        file_name = line["image"]
        instruction = line["text"]
        image_path = os.path.join(self.image_folder, file_name)
        image_tensor = self.image_processor(image_path)
        return instruction, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, image_processor,)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # model
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, processor['image'])

    for (instruct, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]


        pred = mm_infer(
                image_tensor[0],
                instruct[0],
                model=model,
                tokenizer=tokenizer,
                modal='image',
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
            )

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": pred,
                                   "answer_id": ans_id,
                                   # "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample",  action='store_true') 
    args = parser.parse_args()

    eval_model(args)
