import argparse
import glob
import os

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

import shutil
import torch.nn.functional as F


def main(args):

    disable_torch_init()
    
    args.model_path = "models/llava-v1.5-13b"
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

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

    # 检查文件夹是否存在
    gradcam_folder = 'llava_gradcam'
    if os.path.exists(gradcam_folder):
        shutil.rmtree(gradcam_folder)
    os.makedirs(gradcam_folder)

    # 数据集
    data_root = 'dataset/classification'
    # total_datasets = [['species-r' + str(r) for r in range(1, 4)]]
    # total_datasets = [['complex-r' + str(r) for r in range(1, 4)]]
    total_datasets = [['heatmap']]

    # 提问
    prefixs =  ["Answer with the option's letter from the given choices directly. "]
    # prefixs = ['Provide a detailed visual description of the image to answer the following question. ']
    # questions = ['What entity is depicted in the image? (a) {} (b) {}']
    # questions = ['{} (a) {} (b) {}']
    questions = ['How many {} are in the image? (a) {} (b) {}']
    
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
                        image = Image.open(image_file).convert('RGB')
                        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

                        # 将图像张量转换为可学习的参数
                        image_tensor.requires_grad = True

                        conv = conv_templates[args.conv_mode].copy()
                        if "mpt" in model_name.lower():
                            roles = ('user', 'assistant')
                        else:
                            roles = conv.roles

                        challenge = image_file.split('.')[0].split('-')[-3]
                        label = image_file.split('.')[0].split('-')[-2]
                        mislabel = image_file.split('.')[0].split('-')[-1]
                        inp = question.format(challenge, label, mislabel)

                        print(image_file)
                        print("User: " + inp)
                        print(f"{roles[1]}: ", end="")

                        if model.config.mm_use_im_start_end:
                            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                                DEFAULT_IM_END_TOKEN + '\n' + inp
                        else:
                            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        conv.append_message(conv.roles[0], inp)

                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()

                        # 不需要拼接label的操作，因为outputs.logits[0][-1]即是预测的第一个token的输出概率，最大化它即可
                        # prompt = prompt + " There are three men in the image. USER: Solve the problem based on the analysis and image above. Answer with the option's letter from the given choices directly. How many man are in the image? (a) three (b) nine ASSISTANT: "
                        # prompt = prompt + " There are"
                        
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
                            
                        model.train()
                        attention_mask = torch.ones([1, input_ids.size(1)], dtype=torch.bool)
                        inputs = {'input_ids':input_ids, 'labels':targets, 'attention_mask':attention_mask, 'images':image_tensor}
                        outputs = model(**inputs)
                        outputsid = torch.argmax(outputs.logits, dim=-1)
                        tokens = tokenizer.convert_ids_to_tokens(outputsid[0])
                        print(outputsid[0][-4:])
                        print(tokens[-4:])

                        # 非叶子节点保存梯度信息需要在loss.backward()前retain_grad()
                        model.img_emb.retain_grad()
                        (- outputs.logits[0][-1][289] - outputs.logits[0][-1][350]).backward()
                        # (- outputs.logits[0][-1][263] - outputs.logits[0][-1][319]).backward()
                        # (outputs.logits[0][-1][289] + outputs.logits[0][-1][350] - outputs.logits[0][-1][263] - outputs.logits[0][-1][319]).backward() # 350B,289b,319A,263a
                        # (-outputs.logits[0][-1][2211]).backward() # 2211three
                        # outputs.loss.backward()

                        # 将梯度信息聚合到维度为(576)上
                        # gradient_aggregated = model.img_emb.grad.sum(dim=[0, 2])
                        gradient_aggregated = model.img_emb.grad.mean(dim=[0, 2])
                        
                        # 重塑Tensor为(24, 24)
                        reshaped_tensor = gradient_aggregated.view(24, 24)  
                        # 插值到(336, 336)，interpolate函数要求Tensor维度为(minibatch, channels, height, width)，因此需要先增加两个维度
                        reshaped_tensor = reshaped_tensor.unsqueeze(0).unsqueeze(0)  # 现在维度是(1, 1, 24, 24)
                        interpolated_tensor = F.interpolate(reshaped_tensor, size=(336, 336), mode='bilinear', align_corners=False)
                        # 移除额外的维度以用于绘图
                        interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)
                        # 转换为NumPy数组
                        tensor_np = interpolated_tensor.detach().cpu().numpy()
                        # 将Tensor归一化到0-1的范围
                        tensor_min = tensor_np.min()
                        tensor_max = tensor_np.max()
                        normalized_tensor_np = (tensor_np - tensor_min) / (tensor_max - tensor_min)

                        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                        ax[0].imshow(image)
                        ax[0].set_title('Original Image')
                        ax[0].axis('off')  
                        ax[0].set_aspect('equal')
                        ax[1].imshow(image, aspect='equal')
                        gradient_image = ax[1].imshow(normalized_tensor_np, cmap='coolwarm', alpha=0.5, aspect='equal')
                        ax[1].set_title('Gradient Image')
                        ax[1].axis('off')
                        
                        fig.colorbar(gradient_image, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
                        plt.savefig('llava_gradcam/'+image_file.split('/')[-1] + '.png')
                        plt.close()

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