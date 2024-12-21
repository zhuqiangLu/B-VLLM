import os
import re
import json
import argparse
from typing import List, Dict, Optional, Union
from collections import defaultdict

# def extract_characters_regex(s):
#     s = s.strip()
#     answer_prefixes = [
#         "The best answer is",
#         "The correct answer is",
#         "The answer is",
#         "The answer",
#         "The best option is"
#         "The correct option is",
#         "Best answer:"
#         "Best option:",
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, "")

#     if len(s.split()) > 10 and not re.search("[ABCD]", s):
#         return ""
#     matches = re.search(r'[ABCD]', s)
#     if matches is None:
#         return ""
#     return matches[0]


# def eval_your_results(
#         your_results_path: str, 
#         video_types: Optional[Union[List[str], str]] = None,
#         skip_missing: Optional[bool] = True,
#         return_categories_accuracy: Optional[bool] = True,
#         return_sub_categories_accuracy: Optional[bool] = False,
#         return_task_types_accuracy: Optional[bool] = False,
#         gt_answer_key: Optional[str] = "answer",
#         your_answer_key: Optional[str] = "response"

#     ):
#     """
#     Evaluate your results against the ground truth

#     Args:
#     - your_results_path (str): Path to your results file
#     - video_types (Optional[List[str], str]): List of video types to evaluate. 
#     - skip_missing (Optional[bool]): If True, missing files will be skipped. If False, an error will be raised if there are missing files.
#     - return_categories_accuracy (Optional[bool]): If True, the accuracy for each video category will be returned.
#     - return_sub_categories_accuracy (Optional[bool]): If True, the accuracy for each video sub category will be returned.
#     - return_task_types_accuracy (Optional[bool]): If True, the accuracy for each task category will be returned.
#     - gt_answer_key (Optional[str]): Key to access the ground truth answer in the results file.
#     - your_answer_key (Optional[str]): Key to access your answer in the results file.
#     """

#     # Load your results
#     with open(your_results_path, 'r') as f:
#         your_results = json.load(f)

#     if isinstance(video_types, str):
#         video_types = video_types.split(",")

#     q_type_dict = {}
#     v_type_dict = {}
#     v_sub_type_dict = {}


#     for video_type in video_types:

#         # Filter your results based on video types
#         your_results_video_type = [item for item in your_results if item["duration"] == video_type]

#         # Task Categories
#         q_type_dict[video_type] = {}
#         for q_type in TASK_CATEGORIES:
#             q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

#         # Video categories
#         v_type_dict[video_type] = {}
#         for v_type in CATEGORIES:
#             v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}
        
#         v_sub_type_dict[video_type] = {}
#         for v_sub_type in SUB_CATEGORIES:
#             v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

#         if not skip_missing:
#             # Check if the number of files in your results and ground truth are the same
#             assert len(your_results_video_type) == 300, f"Number of files in {video_type} is not 300. Check if there are missing files."

#         for item in your_results_video_type:

#             if skip_missing and item["missing"]:
#                 continue

#             # Get the video category, sub category and question category
#             video_category = item["domain"]
#             video_sub_category = item["sub_category"]
            
#             questions = item["questions"]

#             for question in questions:
#                 q_type = question["task_type"]

#                 # Get the ground truth and your response
#                 gt_answer = question[gt_answer_key]
#                 response = question[your_answer_key]

#                 # Extract the answer from the response
#                 extration = extract_characters_regex(response)
    
#                 if extration != "":
#                     q_type_dict[video_type][q_type]["answered"] += 1
#                     q_type_dict[video_type][q_type]["correct"] += extration == gt_answer

#                     v_type_dict[video_type][video_category]["answered"] += 1
#                     v_type_dict[video_type][video_category]["correct"] += extration == gt_answer

#                     v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
#                     v_sub_type_dict[video_type][video_sub_category]["correct"] += extration == gt_answer


#     # Print the results for each video type
#     for video_type in video_types:

#         print("=====================================")
#         print(f"Evaluation on video Type: {video_type}")
#         print("=====================================")
#         if return_categories_accuracy:
#             print("-------------------------------------")
#             print("Video Domains")
#             print("-------------------------------------")
#             for v_type in v_type_dict[video_type]:
#                 print(f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%")
#         if return_sub_categories_accuracy:
#             print("-------------------------------------")
#             print("Video Sub Categories")
#             print("-------------------------------------")
#             for v_sub_type in v_sub_type_dict[video_type]:
#                 print(f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%")
#         if return_task_types_accuracy:
#             print("-------------------------------------")
#             print("Task Categories")
#             print("-------------------------------------")
#             for q_type in q_type_dict[video_type]:
#                 print(f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%")
        
#         print("-------------------------------------")
#         print("Overall Performance")
#         print("-------------------------------------")
#         total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
#         total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
#         print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

#         print("\n")

#     # Print the results for the entire dataset
#     print("=====================================")
#     print("Evaluation on the entire dataset")
#     print("=====================================")

#     if return_categories_accuracy:
#         print("-------------------------------------")
#         print("Video Categories")
#         print("-------------------------------------")
#         for v_type in CATEGORIES:
#             total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
#             total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
#             print(f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    

#     if return_sub_categories_accuracy:
#         print("-------------------------------------")
#         print("Video Sub Categories")
#         print("-------------------------------------")

#         for v_sub_type in SUB_CATEGORIES:
#             total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
#             total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
#             print(f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")


#     if return_task_types_accuracy:
#         print("-------------------------------------")
#         print("Task Categories")
#         print("-------------------------------------")
#         for q_type in TASK_CATEGORIES:

#             total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
#             total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
#             print(f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

#     print("-------------------------------------")
#     print("Overall Performance")
#     print("-------------------------------------")
#     total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
#     total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
#     print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

# def eval_vnbench(anno_path):
#     # annos = [json.loads(q) for q in open(os.path.expanduser(anno_path), "r")]

#     # annos = json.loads(anno_path)
#     with open(anno_path, 'r') as f:
#         annos = json.load(f)
#     res = defaultdict(list)
#     for anno in annos:
#         label = anno['type']
#         if anno['pred'] is None:
#             continue

#         dic = {
#                 'gt': anno['gt_option'],
#                 'pred': anno['pred'],
#         }
#         if "gpt_judge" in anno:
#             dic['judge'] = anno['gpt_judge'][0]
#         res[label].append(dic)

#     RES = {}
#     result = {}
    
#     for k, vv in res.items():
#         acc = defaultdict(int)
       
#         for v in vv:
#             acc[k] += 1 if v['gt']==v['pred'] else 0
#         # accuracy = 0
#         # for n, ac in acc.items():
#         #     if ac==4:
#         #         accuracy += 1
#         st = f'{k}: true: {acc[k]}, total: {len(vv)}, acc: {acc[k]/len(vv)}'
#         RES[k] = st
#         result[k] = acc[k]/len(vv)
#     RES_list = []
#     for k, v in result.items():
#         print(k)
#         print(RES[k])
#         RES_list.append(result[k])
#     print('Overall: ', sum(RES_list)/len(RES_list))

def eval_vnbench(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)
    res = {}
    for anno in annos:
        name = anno['video']
        label = anno['type']
        if anno['pred'] is None:
            continue
        if not label in res:
            res[label] = []
        if anno['gt'] in [0, 1, 2, 3]:
            anno['gt'] = chr(ord('A') + anno['gt'])
        anno['pred'] = anno['pred'].split('.')[0]
        dic = {
                'name': name,
                'gt': anno['gt_option'],
                'pred': anno['pred'],
        }
        if "gpt_judge" in anno:
            dic['judge'] = anno['gpt_judge'][0]
        res[label].append(dic)

    RES = {}
    result = {}
    sorted_items = sorted(res.items(), key=lambda x: x[0])
    for k, vv in sorted_items:
        acc = {}
        for v in vv:
            name = v['name']
            if not name in acc:
                acc[name] = 0
            if 'judge' in v:
                acc[name] += (v['judge']=='1')
            else:
                pred = v['pred']
                if 'A' in pred:
                    pred = 'A'
                elif 'B' in pred:
                    pred = 'B'
                elif 'C' in pred:
                    pred = 'C'
                elif 'D' in pred:
                    pred = 'D'
                acc[name] += (v['gt']==pred)
        accuracy = 0
        for n, ac in acc.items():
            if ac==4:
                accuracy += 1
        st = f'true: {accuracy}, total: {len(acc)}, acc: {accuracy/len(acc)}'
        RES[k] = st
        result[k] = accuracy/len(acc)
    RES_list = []
    for k, v in result.items():
        # print(k)
        print(f"{k}: {RES[k]}")
        RES_list.append(result[k])
    print('Overall: ', sum(RES_list)/len(RES_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--skip_missing", action="store_true")

    args = parser.parse_args()
    eval_vnbench(args.results_file)

    