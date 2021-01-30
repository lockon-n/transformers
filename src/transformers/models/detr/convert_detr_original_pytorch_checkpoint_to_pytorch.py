# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert DETR checkpoints."""


import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from packaging import version

from PIL import Image
import requests

from transformers import (
    DetrConfig,
    DetrModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):  
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".self_attn.out_proj.weight", "encoder.layers." + str(i) + ".self_attn.out_proj.weight"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".self_attn.out_proj.bias", "encoder.layers." + str(i) + ".self_attn.out_proj.bias"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".linear1.weight", "encoder.layers." + str(i) + ".fc1.weight"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".linear1.bias", "encoder.layers." + str(i) + ".fc1.bias"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".linear2.weight", "encoder.layers." + str(i) + ".fc2.weight"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".linear2.bias", "encoder.layers." + str(i) + ".fc2.bias"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".norm1.weight", "encoder.layers." + str(i) + ".self_attn_layer_norm.weight"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".norm1.bias", "encoder.layers." + str(i) + ".self_attn_layer_norm.bias"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".norm2.weight", "encoder.layers." + str(i) + ".final_layer_norm.weight"))
    rename_keys.append(("transformer.encoder.layers." + str(i) + ".norm2.bias", "encoder.layers." + str(i) + ".final_layer_norm.bias"))
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".self_attn.out_proj.weight", "decoder.layers." + str(i) + ".self_attn.out_proj.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".self_attn.out_proj.bias","decoder.layers." + str(i) + ".self_attn.out_proj.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".multihead_attn.out_proj.weight", "decoder.layers." + str(i) + ".encoder_attn.out_proj.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".multihead_attn.out_proj.bias", "decoder.layers." + str(i) + ".encoder_attn.out_proj.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".linear1.weight", "decoder.layers." + str(i) + ".fc1.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".linear1.bias", "decoder.layers." + str(i) + ".fc1.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".linear2.weight", "decoder.layers." + str(i) + ".fc2.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".linear2.bias", "decoder.layers." + str(i) + ".fc2.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm1.weight", "decoder.layers." + str(i) + ".self_attn_layer_norm.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm1.bias", "decoder.layers." + str(i) + ".self_attn_layer_norm.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm2.weight", "decoder.layers." + str(i) + ".encoder_attn_layer_norm.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm2.bias", "decoder.layers." + str(i) + ".encoder_attn_layer_norm.bias"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm3.weight", "decoder.layers." + str(i) + ".final_layer_norm.weight"))
    rename_keys.append(("transformer.decoder.layers." + str(i) + ".norm3.bias", "decoder.layers." + str(i) + ".final_layer_norm.bias"))


# convolutional projection + query embeddings + layernorm of decoder
rename_keys.extend([("input_proj.weight", "input_projection.weight"),
("input_proj.bias", "input_projection.bias"),
("query_embed.weight", "query_position_embeddings.weight"),
("transformer.decoder.norm.weight", "decoder.layernorm_embedding.weight"),
("transformer.decoder.norm.bias", "decoder.layernorm_embedding.bias")])



def remove_object_detection_heads_(state_dict):
    ignore_keys = [
        "class_embed.weight", 
        "class_embed.bias", 
        "bbox_embed.layers.0.weight", 
        "bbox_embed.layers.0.bias", 
        "bbox_embed.layers.1.weight", 
        "bbox_embed.layers.1.bias", 
        "bbox_embed.layers.2.weight", 
        "bbox_embed.layers.2.bias"
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val

def read_in_q_k_v(state_dict):
    # first: transformer encoder
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop("transformer.encoder.layers." + str(i) + ".self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop("transformer.encoder.layers." + str(i) + ".self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict["encoder.layers." + str(i) + ".self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict["encoder.layers." + str(i) + ".self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict["encoder.layers." + str(i) + ".self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict["encoder.layers." + str(i) + ".self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict["encoder.layers." + str(i) + ".self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict["encoder.layers." + str(i) + ".self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention) 
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention 
        in_proj_weight = state_dict.pop("transformer.decoder.layers." + str(i) + ".self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop("transformer.decoder.layers." + str(i) + ".self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict["decoder.layers." + str(i) + ".self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict["decoder.layers." + str(i) + ".self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict["decoder.layers." + str(i) + ".self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict["decoder.layers." + str(i) + ".self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict["decoder.layers." + str(i) + ".self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict["decoder.layers." + str(i) + ".self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # read in weights + bias of input projection layer of cross-attention 
        in_proj_weight_cross_attn = state_dict.pop("transformer.decoder.layers." + str(i) + ".multihead_attn.in_proj_weight")
        in_proj_bias_cross_attn = state_dict.pop("transformer.decoder.layers." + str(i) + ".multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) of cross-attention to the state dict
        state_dict["decoder.layers." + str(i) + ".encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict["decoder.layers." + str(i) + ".encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict["decoder.layers." + str(i) + ".encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict["decoder.layers." + str(i) + ".encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict["decoder.layers." + str(i) + ".encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict["decoder.layers." + str(i) + ".encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
    

# since we renamed the classification heads of the object detection model, we need to rename the original keys:
rename_keys_object_detection_model = [
("class_embed.weight", "class_labels_classifier.weight"),
("class_embed.bias", "class_labels_classifier.bias"),
("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.weight"),
("bbox_embed.layers.2.weight","bbox_predictor.layers.2.weight"),
("bbox_embed.layers.2.bias","bbox_predictor.layers.2.bias"),
]


@torch.no_grad()
def convert_detr_checkpoint(task, backbone='resnet_50', dilation=False, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    config = DetrConfig()

    if task == "base_model":
        # load model from torch hub
        detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).eval()
        state_dict = detr.state_dict()
        # rename keys
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        # query, key and value matrices need special treatment
        read_in_q_k_v(state_dict)
        # remove classification heads
        remove_object_detection_heads_(state_dict)
        # finally, create model and load state dict
        model = DetrModel(config)
        model.load_state_dict(state_dict)
    
    elif task == "object_detection":
        # load model from torch hub
        if backbone == 'resnet_50' and not dilation:
            detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).eval()
        elif backbone == 'resnet_50' and dilation:
            detr = torch.hub.load('facebookresearch/detr', 'detr_dc5_resnet50', pretrained=True).eval()
        elif backbone == 'resnet_101' and not dilation:
            detr = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True).eval()
        elif backbone == 'resnet_101' and dilation:
            detr = torch.hub.load('facebookresearch/detr', 'detr_dc5_resnet101', pretrained=True).eval()
        else: 
            print("Not supported:", backbone, dilation)
        
        state_dict = detr.state_dict()
        # rename keys
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        # query, key and value matrices need special treatment
        read_in_q_k_v(state_dict)
        # rename classification heads
        for src, dest in rename_keys_object_detection_model:
            rename_key(state_dict, src, dest)
        # finally, create model and load state dict
        model = DetrForObjectDetection(config)
        model.load_state_dict(state_dict)
    elif task == "panoptic_segmentation":
        # First, load in original detr from torch hub
        if backbone == 'resnet_50' and not dilation:
            detr, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet50_panoptic', 
                                                pretrained=True, return_postprocessor=True, num_classes=250)
            detr.eval()
        elif backbone == 'resnet_50' and dilation:
            detr, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_dc5_resnet50_panoptic', 
                                                pretrained=True, return_postprocessor=True, num_classes=250)
            detr.eval()
        elif backbone == 'resnet_101' and not dilation:
            detr, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', 
                                                pretrained=True, return_postprocessor=True, num_classes=250)
            detr.eval()
        else:
            print("Not supported:", backbone, dilation)

    else:
        print("Task not in list of supported tasks:", task)

    # Check results on an image of cute cats
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # verify outputs
    if task == "base_model":
        assert outputs.last_hidden_state.shape == (1, 100, 256)
        assert outputs.shape == outputs.shape
        #assert (original_output == outputs).all().item()
    elif task == "object_detection":
        raise NotImplementedError
    elif task == "panoptic_segmenation":
        raise NotImplementedError
    
    # Save model
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("task", default='base_model', type=str, help="""Task for which to convert a checkpoint. One of 'base_model', 
    # 'object_detection' or 'panoptic_segmentation'. """)
    # parser.add_argument("backbone", default='resnet_50', type=str, help="Which backbone to use. One of 'resnet50', 'resnet101'.")
    # parser.add_argument("dilation", default=False, action="store_true", help="Whether to apply dilated convolution.")
    # parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # args = parser.parse_args()
    # convert_detr_checkpoint(args.task, args.backbone, args.dilation, args.pytorch_dump_folder_path)
    convert_detr_checkpoint(task='base_model')