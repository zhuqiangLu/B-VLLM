# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np

import einops
import torch
import torch.nn as nn

from .projector import load_mm_projector, build_vision_projector
# from .encoder import build_vision_tower
from .eva_vit import build_vision_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP
from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw
import json
from einops import einsum, rearrange
import math

from .merge import threshold_soft_matching, merge_wavg, multi_layer_merge

def merge_temporal(merge_metrics, mm_img_in):
    # threshold = random.uniform(0.5, 0.9)
    threshold = 0.75
    metrics = threshold_soft_matching(merge_metrics, threshold=threshold)
    n = mm_img_in.shape[1]
    c = mm_img_in.shape[2]
    mm_img_in = rearrange(mm_img_in, 'k n c -> k (n c)')
    merged_mm_img_in, _ = merge_wavg(metrics, mm_img_in)
    merged_mm_img_in = rearrange(merged_mm_img_in, 'k (n c) -> k n c',n=n, c=c)
    # print(merged_mm_img_in.shape)
    return merged_mm_img_in

class BVLMMetaModel:

    def __init__(self, config):
        super(BVLMMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

       

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)

    def initialize_spatial_attention_modules(self, model_args, for_eval=False, vis_device='cuda'):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.bert_type = getattr(model_args, "bert_type", "qformer")
        self.config.num_query = getattr(model_args, "num_query", 32)
        self.config.compress_type = getattr(model_args, "compress_type", None)

        if 'pretrain' in self.config.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.config.mm_hidden_size

        
        self.vlm_att_tokenlizer_spatial, self.vlm_att_encoder_spatial, self.vlm_att_query_spatial = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector_spatial = torch.nn.Linear(self.vlm_att_encoder_spatial.config.hidden_size, self.config.mm_hidden_size)
        self.vlm_att_key_projector_spatial  = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)
        self.vlm_att_val_projector_spatial  = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if "raw" in self.config.bert_type:
            self.vlm_att_bert_proj_spatial  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder_spatial.config.hidden_size)
        elif "pretrain" in self.config.bert_type and self.config.mm_hidden_size!=att_feat_size:
            self.vlm_att_bert_proj_spatial = torch.nn.Linear(self.config.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj_spatial = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'qformer_pretrain' in self.config.bert_type:
            self.vlm_att_ln_spatial = torch.nn.LayerNorm(att_feat_size)
        
        # print(self.vlm_att_encoder_spatial)
        if pretrain_qformer is not None:
            print("Loading pretrained qformer weights...")
            
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            # print(self.vlm_att_encoder_spatial.bert.encoder.layer[0].output_query.dense.weight.shape)
            self.vlm_att_encoder_spatial.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln_spatial.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query_spatial.data = qformer_weight['query_tokens']
        
        if 'freeze_all' in self.config.bert_type:
            print("Freezing all qformer weights...")
            self.vlm_att_encoder_spatial.requires_grad_(False)
            self.vlm_att_ln_spatial.requires_grad_(False)
            self.vlm_att_query_spatial.requires_grad_(False)
            self.vlm_att_projector_spatial.requires_grad_(False)
            self.vlm_att_key_projector_spatial.requires_grad_(False)
            self.vlm_att_val_projector_spatial.requires_grad_(False)
        elif 'freeze' in self.config.bert_type:
            print("Freezing pretrained qformer weights...")
            self.vlm_att_encoder_spatial.requires_grad_(False)
            self.vlm_att_ln_spatial.requires_grad_(False)
            self.vlm_att_query_spatial.requires_grad_(False)
        
        att_projector_weights = {}
        if pretrain_mm_mlp_adapter is not None:
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            return

        if "qformer" in self.config.bert_type:
            print('Loading vlm_att_query_spatial weights...')
            self.vlm_att_query_spatial.data = att_projector_weights['model.vlm_att_query_spatial']
            if "pretrain" in self.config.bert_type:
                print('Loading vlm_att_ln weights...')
                self.vlm_att_ln_spatial.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln_spatial'))

        if self.vlm_att_bert_proj_spatial is not None:
            print('Loading vlm_att_bert_proj weights...')
            self.vlm_att_bert_proj_spatial.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj_spatial'))
        
        


    def initialize_temporal_attention_modules(self, model_args, for_eval=False, vis_device='cuda'):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.bert_type = getattr(model_args, "bert_type", "qformer")
        self.config.num_query = getattr(model_args, "num_query", 32)
        self.config.compress_type = getattr(model_args, "compress_type", None)

        if 'pretrain' in self.config.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.config.mm_hidden_size
        self.vlm_att_tokenlizer_temporal, self.vlm_att_encoder_temporal, self.vlm_att_query_temporal = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector_temporal = torch.nn.Linear(self.vlm_att_encoder_temporal.config.hidden_size, self.config.mm_hidden_size)
        self.vlm_att_key_projector_temporal  = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)
        self.vlm_att_val_projector_temporal  = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if "raw" in self.config.bert_type:
            self.vlm_att_bert_proj_temporal  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder_temporal.config.hidden_size)
        elif "pretrain" in self.config.bert_type and self.config.mm_hidden_size!=att_feat_size:
            self.vlm_att_bert_proj_temporal = torch.nn.Linear(self.config.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj_temporal = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'qformer_pretrain' in self.config.bert_type:
            self.vlm_att_ln_temporal = torch.nn.LayerNorm(att_feat_size)
        
        if pretrain_qformer is not None:
            print("Loading pretrained qformer weights...")
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            self.vlm_att_encoder_temporal.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln_temporal.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query_temporal.data = qformer_weight['query_tokens']
        
       

        att_projector_weights = {}
        if pretrain_mm_mlp_adapter is not None:
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            return 

        if "qformer" in self.config.bert_type:
            print('Loading vlm_att_query weights...')
            self.vlm_att_query_temporal.data = att_projector_weights['model.vlm_att_query_temporal']
            if "pretrain" in self.config.bert_type:
                print('Loading vlm_att_ln weights...')
                self.vlm_att_ln_temporal.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln_temporal'))

        if self.vlm_att_bert_proj_temporal is not None:
            print('Loading vlm_att_bert_proj weights...')
            self.vlm_att_bert_proj_temporal.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj_temporal'))
    

    def post_loading(self):

        weight_type = torch.float16
        device_type = self.mm_projector.readout[0].weight.device

        self.vlm_att_encoder_spatial = self.vlm_att_encoder_spatial.to(device=device_type, dtype=weight_type)
        self.vlm_att_projector_spatial = self.vlm_att_projector_spatial.to(device=device_type, dtype=weight_type)
        self.vlm_att_key_projector_spatial = self.vlm_att_key_projector_spatial.to(device=device_type, dtype=weight_type)
        self.vlm_att_val_projector_spatial = self.vlm_att_val_projector_spatial.to(device=device_type, dtype=weight_type)

        if "qformer" in self.config.bert_type:
            self.vlm_att_query_spatial.data = self.vlm_att_query_spatial.data.to(device=device_type, dtype=weight_type)
            if "pretrain" in self.config.bert_type:
                self.vlm_att_ln_spatial = self.vlm_att_ln_spatial.to(device=device_type, dtype=weight_type)
        
        if self.vlm_att_bert_proj_spatial is not None:
            self.vlm_att_bert_proj_spatial = self.vlm_att_bert_proj_spatial.to(device=device_type, dtype=weight_type)
        
        self.vlm_att_encoder_temporal = self.vlm_att_encoder_temporal.to(device=device_type, dtype=weight_type)
        self.vlm_att_projector_temporal = self.vlm_att_projector_temporal.to(device=device_type, dtype=weight_type)
        self.vlm_att_key_projector_temporal = self.vlm_att_key_projector_temporal.to(device=device_type, dtype=weight_type)
        self.vlm_att_val_projector_temporal = self.vlm_att_val_projector_temporal.to(device=device_type, dtype=weight_type)

        if "qformer" in self.config.bert_type:
            self.vlm_att_query_temporal.data = self.vlm_att_query_temporal.data.to(device=device_type, dtype=weight_type)
            if "pretrain" in self.config.bert_type:
                self.vlm_att_ln_temporal = self.vlm_att_ln_temporal.to(device=device_type, dtype=weight_type)
        
        if self.vlm_att_bert_proj_temporal is not None:
            self.vlm_att_bert_proj_temporal = self.vlm_att_bert_proj_temporal.to(device=device_type, dtype=weight_type)


    def initialize_attention_modules(self, model_args, for_eval=False):  
        self.initialize_spatial_attention_modules(model_args, for_eval)
        self.initialize_temporal_attention_modules(model_args, for_eval)

    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        query_tokens = None

        # a = torch.nn.Linear(10, 10)
        # print(encoder_config)
        
        if "qformer" in self.config.bert_type:
            mm_model = BertLMHeadModelQF(encoder_config)
            query_tokens = nn.Parameter(
                torch.zeros(1, self.config.num_query, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            # print(mm_model.bert.encoder.layer[0].output_query.dense.weight.shape)
            # raise
        elif "raw" in self.config.bert_type:
            encoder_config.is_decoder = True
            mm_model = BertLMHeadModelRaw.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
        else:
            raise NotImplementedError("BERT type not implemented...")
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        if "layer" in self.config.bert_type:
            layer_num = int(self.config.bert_type.split(':')[-1])
            mm_model.bert.encoder.layer = mm_model.bert.encoder.layer[:layer_num]
            print(f"Only use {layer_num} layers in BERT...")
        # print(mm_model.bert.encoder)

        return tokenizer, mm_model, query_tokens


class BVLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()




    def vlm_attention(self, image_features, prompts=None, image_counts=None, long_video=False):        
        img_feat_lst = []
        device_type = self.get_model().mm_projector.readout[0].weight.device
        image_features = image_features.to(device_type)
        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(device_type)    

        total_count = 0
        # calculate each image feat according to the prompt
        for _idx in range(len(prompts)):
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            input_token = self.get_model().vlm_att_tokenlizer_temporal(
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(device_type)

            input_ids = input_token.input_ids
            attention_masks = input_token.attention_mask
            

            if image_counts is None:
                img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1, -1)
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)
            else:
                if image_counts[_idx] > 1:
                    expand_size = 1
                else:
                    expand_size = len(prompts[_idx])
                    # shape: [prompt_num*frame_num, image_shape, feat_dim]
                img_feat_prompt = image_features[total_count:total_count+image_counts[_idx]]
                img_feat_prompt = img_feat_prompt[None].expand(expand_size, -1, -1, -1).flatten(0,1)
                img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(expand_size, -1, -1).flatten(0,1)
                input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                total_count += image_counts[_idx]
        
            # print(image_counts[_idx], len(prompts[_idx]))

            # remove cls embedding
            cls_tokens = None

            if self.config.mm_vision_select_feature == 'patch':
                if img_feat_prompt.shape[1]%2 == 1:
                    cls_tokens = img_feat_prompt[:, :1]
                    img_feat_prompt = img_feat_prompt[:, 1:]
                else:
                    cls_tokens = img_feat_prompt.mean(dim=1, keepdim=True)

            if "pretrain" in self.config.bert_type and self.get_model().vlm_att_bert_proj_temporal is not None:
                bert_feat = self.get_model().vlm_att_bert_proj_temporal(img_feat_prompt)
            else:
                bert_feat = img_feat_prompt.clone()
            N = 32
            if "qformer" in self.config.bert_type:
                query_tokens = self.get_model().vlm_att_query_spatial.expand(bert_feat.shape[0], -1, -1)
                query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device), 
                                        attention_masks],dim=1)

                # print('sss ', query_tokens.shape, query_atts.shape)
                if 'pretrain' in self.config.bert_type:
                    mm_img_in = self.get_model().vlm_att_ln_temporal(bert_feat)
                else:
                    mm_img_in = bert_feat

                # temporal sampling 
                temporal_query = self.get_model().vlm_att_query_temporal[:, :16].expand(cls_tokens.shape[0], -1, -1)
                tmp_query_atts = torch.cat([torch.ones(temporal_query.size()[:-1], dtype=torch.long).to(cls_tokens.device), 
                                        attention_masks],dim=1)
                # temporal selector, only for video, we know video vqa is only one round in this dataset
                if image_counts is not None and image_counts[_idx] > 1:
                    
                    cls_tokens = rearrange(cls_tokens, 'b n c -> (b n) c').unsqueeze(dim=0)
                    cls_attn = torch.ones((1, cls_tokens.shape[1]), dtype=cls_tokens.dtype, device=cls_tokens.device)
                    # print(input_ids.device, temporal_query.device, cls_tokens.device, cls_attn.device, tmp_query_atts.device, self.get_model().vlm_att_encoder_temporal.device)

                    temporal_selector = self.get_model().vlm_att_encoder_temporal.bert(
                        input_ids[:1],
                        query_embeds=temporal_query[:1],
                        attention_mask=tmp_query_atts[:1],
                        encoder_hidden_states=cls_tokens,
                        encoder_attention_mask=cls_attn,
                        return_dict=True,
                    )
                    temporal_selector = temporal_selector.last_hidden_state[:,:temporal_query.shape[1]]
                    # temporal_selector = temporal_selector.last_hidden_state[:,:3]
                    temporal_selector = self.get_model().vlm_att_projector_temporal(temporal_selector)# 1 32 C
                    temporal_selector = einsum(temporal_selector, cls_tokens, "b k c, b T c -> b k T")
                    c = cls_tokens.shape[-1]

                    # here, we use gumble softmax to control the ordering issue
                    temporal_selector = torch.nn.functional.gumbel_softmax(temporal_selector/math.sqrt(c), dim=-1, tau=0.5)
                    temporal_selector = temporal_selector.squeeze(dim=0)

                    # there are a couple of options 

                    self.temporal_selector = temporal_selector.detach()
                    mm_img_in = einsum(temporal_selector, mm_img_in, 'k T, T n c -> k n c')#[sorted_query_index]
                    mm_img_in = merge_temporal(temporal_selector, mm_img_in)
                    
                 
                    max_T = 576 // N 
                    # max_T = 256 // N 
                    cur_T = mm_img_in.shape[0]
                    N = max(1, (max_T/cur_T)*N)
                    N = int(N)

                    k = mm_img_in.shape[0]
                    input_ids = input_ids[:1].repeat(k, 1)
                    query_tokens = query_tokens[:1].repeat(k, 1, 1)
                    query_atts = query_atts[:1].repeat(k, 1)
                    img_att_prompt = img_att_prompt[:1].repeat(k, 1)

                else:
                    cls_attn = torch.ones((cls_tokens.shape[0], cls_tokens.shape[1]), dtype=cls_tokens.dtype, device=cls_tokens.device)

                    temporal_selector = self.get_model().vlm_att_encoder_temporal.bert(
                        input_ids,
                        query_embeds=temporal_query,
                        attention_mask=tmp_query_atts,
                        encoder_hidden_states=cls_tokens,
                        encoder_attention_mask=cls_attn,
                        return_dict=True,
                    )

                    temporal_selector = temporal_selector.last_hidden_state[:,:temporal_query.shape[1]]
                    temporal_selector = self.get_model().vlm_att_projector_temporal(temporal_selector)# 1 32 C
                    temporal_selector = einsum(temporal_selector, cls_tokens, "b k c, b T c -> b k T")

                    c = cls_tokens.shape[-1]

                    # here, we use gumble softmax to control the ordering issue
                    # temporal_selector = torch.nn.functional.softmax(temporal_selector/math.sqrt(c), dim=-1)
                    temporal_selector = torch.nn.functional.gumbel_softmax(temporal_selector/math.sqrt(c), dim=-1, tau=0.5)
            
                    mm_img_in = einsum(temporal_selector, mm_img_in.unsqueeze(dim=1), 'b k T, b T n c -> b k n c').mean(dim=1)#[sorted_query_index]

                if long_video:
                    outputs = []
                    block_size = 64
                    for L in range(0, len(input_ids), block_size):
                        R = L + block_size
                        mm_output = self.get_model().vlm_att_encoder_spatial.bert(
                            input_ids[L:R],
                            query_embeds=query_tokens[L:R],
                            attention_mask=query_atts[L:R],
                            encoder_hidden_states=mm_img_in[L:R],
                            encoder_attention_mask=img_att_prompt[L:R],
                            return_dict=True,
                        )
                        mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                        outputs.append(mm_output)
                    mm_output = torch.cat(outputs)
                    torch.cuda.empty_cache()
                else:
                    # print(mm_img_in.shape, query_tokens.shape, query_atts.shape, img_att_prompt.shape)
                    mm_output = self.get_model().vlm_att_encoder_spatial.bert(
                        input_ids,
                        query_embeds=query_tokens,
                        attention_mask=query_atts,
                        encoder_hidden_states=mm_img_in,
                        encoder_attention_mask=img_att_prompt[:, :-1], 
                        return_dict=True,
                        output_attentions=True,
                    )
                    mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                mm_output = multi_layer_merge(mm_output, N)
            
            
            else:
                raise ValueError(f'Unexpected bert type: {self.config.bert_type}')
            
            final_token = self.get_model().vlm_att_projector_spatial(mm_output)

            if image_counts is not None:
                # shape: [prompt_num, frame_num*image_shape, feat_dim]
                final_token = final_token.reshape(len(prompts[_idx]), -1, *final_token.shape[-2:])
                # print(final_token.shape)
                final_token = final_token.flatten(1,2)
                # print(final_token.shape)

            final_token = self.get_model().vlm_att_val_projector_spatial(final_token).squeeze(dim=0)

            
            
            img_feat_lst.append(final_token)

        return img_feat_lst

        

    def encode_images_or_videos(self, images, prompts=None):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = list()
        image_counts = list() 


        for i, (data, modal) in enumerate(images):
            # if model == 'image':
            image_counts.append(data.shape[0])
            data_batch.append(data)
            
        data_batch = torch.cat(data_batch, dim=0)

        # assert len(data_batch.size()) == 5
        # batch_size = data_batch.size(0)

        # frames = einops.rearrange(data_batch, 'b t c h w -> (b t) c h w')
        if len(data_batch.shape) == 4:
            frames_features = self.get_model().get_vision_tower()(data_batch)
        else:
            frames_features = data_batch
        frames_features = self.vlm_attention(frames_features, prompts, image_counts)

        return frames_features

    def temporal_aggregator(self, frames_features):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            # video_features = self.get_model().mm_projector(frames_features.mean(1))
            video_features = self.get_model().mm_projector(frames_features.mean)
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")

        return video_features

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None, sep_id=None,
    ):
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        

        mm_features = self.encode_images_or_videos(images, prompts)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            if sep_id is not None:
                sep_feat = self.get_model().embed_tokens(sep_id)
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_mm_idx += 1
                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
