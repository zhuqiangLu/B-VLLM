
<h3 align="center"> B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens</a></h3>


[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/LICENSE) 
[![arXiv](https://img.shields.io/badge/Arxiv-2412.09919-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2412.09919) <br>



## Installation 
```
git clone https://github.com/zhuqiangLu/B-VLLM.git
cd B-VLLM
conda create -n bvllm python==3.10
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
```

## Data Preparation 
Here, we utilize the [Video-LLaVA dataset](https://huggingface.co/datasets/LanguageBind/Video-LLaVA) to train our model. Note that we follow the training receipt provided by [Video-LLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) to train our model.
Once the dataset is downloaded, organize the them as follow:
```
B-VLLM
├── datasets
│   ├── videollava_pt
|   |   ├── llava_image/ 
|   |   ├── valley/     
|   |   └── valley_llavaimage.json # Available at: https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view, including 703K video-text and 558K image-text pairs
│   ├── videollava_sft
|   |   ├── llava_image_tune/  
|   |   ├── videochatgpt_tune/ 
|   |   └── videochatgpt_llavaimage_tune.json # Available at: https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view, including 100K video-centric, 625K image-centric and 40K text-only conversations
```

## Training 
Make sure you update the `ARG_NPROC_PER_NODE` to match your available GPUs before running the script.
```
bash scripts/vllava/pretrain.sh
bash scripts/vllava/finetune.sh
```

## Evaluation 
Comning Soon

