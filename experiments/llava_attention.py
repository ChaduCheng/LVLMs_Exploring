import argparse
import glob
import os
from sys import prefix

import torch
from PIL import Image
from transformers import TextStreamer

from llava.constants import (IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

import matplotlib.pyplot as plt
import numpy as np
import re

import shutil
import torch.nn.functional as F


def main(args):

    disable_torch_init()
    
    attention_img_folder = "./attention_images"
    if os.path.exists(attention_img_folder):
        shutil.rmtree(attention_img_folder)
    os.makedirs(attention_img_folder)
    
    args.model_path = "models/llava-v1.5-13b"
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    # 为了让梯度能够流回输入图像的每个像素上
    for param in model.parameters():
        param.requires_grad = True
    # 计算完一张图片的梯度后需要清除模型梯度
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
            conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # 数据集
    data_root = 'dataset/classification'
    total_datasets = [['attention']]

    # 提问
    prefixs =  ["Answer with the option's letter from the given choices directly. "]
    # prefixs = ['Provide a detailed visual description of the image to answer the following question. ']
    
    # questions = ['What entity is depicted in the image? (a) {} (b) {}']
    # questions = ['{} (a) {} (b) {}']
    questions = ['How many {} are in the image? (a) {} (b) {}']
    
    chosen_image = "11-man-three-nine.jpg"
    
    for q in questions:
        for p in prefixs:
            question = p + q
            for datasets in total_datasets:
                for dataset in datasets:
                    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
                    image_files = []
                    for extension in image_extensions:
                        image_files.extend(glob.glob(os.path.join(data_root, dataset, extension)))

                    for image_file in image_files:
                        if chosen_image not in image_file:
                            continue
                        
                        image = Image.open(image_file).convert('RGB')
                        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

                        conv = conv_templates[args.conv_mode].copy()
                        if "mpt" in model_name.lower():
                            roles = ('user', 'assistant')
                        else:
                            roles = conv.roles
                        
                        inp = None
                        if len(re.findall(r'\{\}', question)) == 3:
                            challenge = image_file.split('.')[0].split('-')[-3]
                            label = image_file.split('.')[0].split('-')[-2]
                            mislabel = image_file.split('.')[0].split('-')[-1]
                            inp = question.format(challenge, label, mislabel)
                        else:
                            label = image_file.split('.')[0].split('-')[-2]
                            mislabel = image_file.split('.')[0].split('-')[-1]
                            inp = question.format(label, mislabel)

                        print(image_file)
                        print("User: " + inp)
                        print(f"{roles[1]}: ", end="")

                        if model.config.mm_use_im_start_end:
                            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                        else:
                            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        conv.append_message(conv.roles[0], inp)

                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()

                        # 不需要拼接label的操作，因为outputs.logits[0][-1]即是预测的第一个token的输出概率，最大化它即可
                        # 这里只是为了查看attention才进行拼接
                        prompt = prompt + " B"
                        # prompt = prompt + " The image depicts a flamingo, not a Komodo dragon, as it is standing in the water with its wings spread out. USER: Solve the problem based on the analysis above. Answer with the option's letter from the given choices directly. What entity is depicted in the image? (a) flamingo (b) Komodo dragon ASSISTANT: A"
                        # prompt = prompt + " There are three men in the image. USER: Solve the problem based on the analysis and image above. Answer with the option's letter from the given choices directly. How many man are in the image? (a) three (b) nine ASSISTANT: A"
                        
                        # Mask targets
                        conversations = [prompt]
                        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                        targets = input_ids.clone()
                        sep = conv.sep + conv.roles[1] + ": "
                        for conversation, target in zip(conversations, targets):
                            total_len = int(target.ne(tokenizer.pad_token_id).sum())
                            rounds = conversation.split(conv.sep2)
                            cur_len = 1
                            target[:cur_len] = IGNORE_INDEX
                            for i, rou in enumerate(rounds):
                                if rou == "":
                                    break
                                parts = rou.split(sep)
                                if len(parts) != 2:
                                    break
                                parts[0] += sep
                                round_len = len(tokenizer_image_token(rou, tokenizer))
                                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                                cur_len += round_len
                            target[cur_len:] = IGNORE_INDEX
                        
                        attention_mask = torch.ones([1, input_ids.size(1)], dtype=torch.bool)
                        inputs = {'input_ids':input_ids, 'labels':targets, 'attention_mask':attention_mask, 'images':image_tensor, 'output_attentions':True, 'output_hidden_states':True}
                        outputs = model(**inputs)
                        outputsid = torch.argmax(outputs.logits, dim=-1)
                        tokens = tokenizer.convert_ids_to_tokens(outputsid[0])
                        
                        print()
                        topk = 10
                        layers_num = 40
                        for layer in range(layers_num):
                            print(f"Layer - {layer} - top{topk} tokens")
                            # 层注意力
                            attention_weights = outputs['attentions']
                            layer_attention = attention_weights[layer]  
                            # 沿头维度平均
                            head_averaged_attention = layer_attention.mean(dim=1).squeeze(dim=0)
                            
                            # 计算图像嵌入部分的平均关注度
                            image_embed_start = 35
                            image_embed_end = 35 + 576
                            # image_embed_attention_avg = head_averaged_attention[:, image_embed_start:image_embed_end].mean(dim=1, keepdim=True)
                            image_embed_attention_avg = head_averaged_attention[:, image_embed_start:image_embed_end].sum(dim=1, keepdim=True)
                            
                            # 提取非图像嵌入部分的关注度
                            non_image_attention_before = head_averaged_attention[:, :image_embed_start]
                            non_image_attention_after = head_averaged_attention[:, image_embed_end:]
                            # 按顺序拼接三部分：图像嵌入部分之前的关注度、图像嵌入的平均关注度、图像嵌入部分之后的关注度
                            head_averaged_attention = torch.cat((non_image_attention_before, image_embed_attention_avg, non_image_attention_after), dim=1)
                            
                            # 将图片token等的注意力置0, 便于其它token可视化
                            head_averaged_attention[:, 35] = 0.0
                            head_averaged_attention[:, 26] = 0.0
                            head_averaged_attention[:, 0] = 0.0
                            head_averaged_attention[:, :36] = 0.0
                            # 移除 image embedding 的576行
                            head_averaged_attention = torch.cat((head_averaged_attention[:35,:], head_averaged_attention[611:,:]), dim=0)
                            
                            head_averaged_attention_softmax = F.softmax(head_averaged_attention, dim=-1)
                            head_averaged_attention_np = head_averaged_attention_softmax.detach().cpu().numpy()

                            plt.figure(figsize=(10, 8))
                            plt.imshow(head_averaged_attention_np, cmap='viridis', aspect='auto')
                            plt.colorbar()  
                            plt.title("")
                            plt.xlabel("")
                            plt.ylabel("")
                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20)
                            plt.savefig(os.path.join(attention_img_folder, f"average_attention_weights_layer_{layer}.png"), bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                            # 最终输出选项时 topk 最大关注度的token
                            head_averaged_attention_softmax = head_averaged_attention_softmax[-1:, :]
                            _, top_indices = torch.topk(head_averaged_attention_softmax, topk, dim=-1)
                            top_indices = top_indices.squeeze()
                            for i in top_indices:
                                index = i.item()
                                if index <= 36 or index >= len(input_ids[0]) - 1: # 不计入图片token和最后的选项token
                                    continue
                                else:
                                    pre_token = tokenizer.convert_ids_to_tokens((input_ids[0][index-1]).item())
                                    post_token = tokenizer.convert_ids_to_tokens((input_ids[0][index+1]).item())
                                    curr_token = tokenizer.convert_ids_to_tokens((input_ids[0][index]).item())
                                    print(pre_token, curr_token, post_token)
                            print()
                        
                        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    main(args)