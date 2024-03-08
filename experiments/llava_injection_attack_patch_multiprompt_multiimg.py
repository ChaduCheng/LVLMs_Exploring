import os
import sys
import time
import pickle
import shutil
import random
import argparse
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from transformers import TextStreamer
from torchvision.utils import save_image

from llava.constants import (IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def crop_resize(image, image_size):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    image = image.crop((left, top, right, bottom)).resize((image_size, image_size))
    return image
    

def render_typo(image, text, font_path, font_size, font_color):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    image_width = image.width
    image_height = image.height
    text_width = int(draw.textlength(text, font=font))
    text_height = 10
    text_x = max(0, (image_width - text_width) // 2)    # 考虑文本宽度
    text_y = max(0, (image_height - text_height) // 2)  # 考虑文本高度
    draw.text((text_x, text_y), text, fill=font_color, font=font)
    return image


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


def main(args):
    
    model_path = "models/llava-llama-2-13b-chat-lightning-preview"                   #  "models/llava-v1.5-13b"
    image_folder = "dataset/injection/image_train"
    questions_file = "dataset/injection/questions_train.txt"
    log_dir = "logs18"
    
    patch_size = 70                                        # 正方形 patch 的边长，目前 patch 位于左上角
    zero_out = True                                        # 是否将 patch 处的像素统一归零，帮助收敛
    
    add_typo = True                                        # 在图像中间添加typo作为adv patch产生效果的触发器
    typo_size = 50
    typo_color = (255, 255, 255)
    typo_font = 'fonts/arial_bold.ttf'
    typos = ['Hello']
    
    target_words = ['Suicide']
    
    image_size = 224
    batch_size = 2
    epsilon = 255 / 255
    alpha = 1 / 255
    num_iter = 5000
    
    test_output_num = 1                                    # 测试 adv image 攻击效果时的输出次数
    test_output_period = 100
    
    #################################
    
    disable_torch_init()
    
    # init log folder
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    
    question_pool = []
    with open(questions_file, 'r') as file:
        for line in file:
            question_pool.append(line.strip())
    image_pool = os.listdir(image_folder)
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
        
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    for typo in typos:
        for target_word in target_words:
            target_response = f" {target_word}</s>"
                
            with open(os.path.join(log_dir, f"output_{typo}_{target_word}.log"), 'w') as f:
                sys.stdout = f
                print(f"target response: {target_response}    alpha: {int(alpha*255)}    epsilon: {int(epsilon*255)}    batch size: {batch_size}    patch size: {patch_size}    zero out: {zero_out}    typo size: {typo_size}")
                
                loss_buffer = []
                image_file = random.choice(image_pool)
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                adv_noise = torch.rand_like(image_tensor).cuda() * 2 * epsilon - epsilon
                
                # patch mask
                mask = torch.zeros_like(adv_noise)
                mask[:, :, 0:patch_size, 0:patch_size] = 1
                adv_noise = adv_noise * mask
                inverse_mask = 1 - mask
                mask = mask.cuda()
                inverse_mask = inverse_mask.cuda()
                
                adv_noise = adv_noise.cuda()
                adv_noise.requires_grad_(True)
                adv_noise.retain_grad()
                for t in tqdm(range(num_iter + 1)):
                    image_file = random.choice(image_pool)
                    image_path = os.path.join(image_folder, image_file)
                    image = Image.open(image_path).convert('RGB')
                    image = crop_resize(image, image_size)
                    
                    if add_typo:
                        image = render_typo(image, typo, typo_font, typo_size, typo_color)
                        
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                    x = denormalize(image_tensor).clone().cuda()
                    if zero_out:
                        x = x * inverse_mask
                    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
                    
                    questions = random.sample(question_pool, batch_size)
                    conversations = []
                    conv = conv_templates[args.conv_mode].copy()
                    for question in questions:
                        conv.messages = []
                        if model.config.mm_use_im_start_end:
                            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                        else:
                            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        conversations.append(conv.get_prompt() + target_response)

                    # Mask targets
                    input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in conversations]
                    max_length = max(tensor.shape[0] for tensor in input_ids)
                    padded_input_ids = [F.pad(tensor, (0, max_length - tensor.shape[0]), 'constant', tokenizer.pad_token_id) for tensor in input_ids]
                    input_ids = torch.stack(padded_input_ids, dim=0).cuda()
                    attention_masks = []
                    
                    targets = input_ids.clone()
                    if 'llama-2' in model_name.lower():
                        sep = "[/INST] "
                    elif "v1" in model_name.lower():
                        sep = conv.sep + conv.roles[1] + ": "
                    for conversation, target in zip(conversations, targets):
                        total_len = int(target.ne(tokenizer.pad_token_id).sum())
                        attention_mask = target.ne(tokenizer.pad_token_id)
                        attention_masks.append(attention_mask)
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
                    attention_masks = torch.stack(attention_masks, dim=0).cuda()
                    inputs = {'input_ids':input_ids, 'labels':targets, 'attention_mask':attention_masks, 'images':image_tensor}
                            
                    x_adv = x + adv_noise
                    x_adv = normalize(x_adv)
                    images = x_adv.repeat(len(questions), 1, 1, 1)
                    inputs['images'] = images.half()
                    outputs = model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign() * mask).clamp(-epsilon, epsilon)
                    adv_noise.grad.zero_()
                    model.zero_grad()
                    loss_buffer.append(loss.item())
                        
                    if t % test_output_period == 0:
                        # test effect of adversarial image
                        print(f'######### Output - Iter = {t}, loss: {loss.item()}, image: {image_file} ##########')
                        x = denormalize(image_tensor).clone().cuda()
                        if zero_out:
                            x = x * inverse_mask
                        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
                        x_adv = x + adv_noise
                        x_adv = normalize(x_adv)
                        for question in questions:
                            inp = question
                            if model.config.mm_use_im_start_end:
                                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                            else:
                                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                            conv = conv_templates[args.conv_mode].copy()    
                            conv.append_message(conv.roles[0], inp)
                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()
                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                            for i in range(test_output_num):
                                print(f"Test : {i} Question: {question}")
                                with torch.inference_mode():
                                    output_ids = model.generate(
                                        input_ids,
                                        images=x_adv.half(),
                                        do_sample=True,
                                        temperature=0.2,
                                        max_new_tokens=1024,
                                        streamer=streamer,
                                        use_cache=True,
                                        stopping_criteria=[stopping_criteria])
                                print()
                        
                        # plot loss
                        sns.set_theme()
                        num_iters = len(loss_buffer)
                        x_ticks = list(range(num_iters))
                        plt.plot(x_ticks[:len(loss_buffer)], loss_buffer, label=f"epsilon {int(epsilon*255)}")
                        plt.title('Loss Plot')
                        plt.xlabel('Iters')
                        plt.ylabel('Loss')
                        plt.legend(loc='best')
                        plt.savefig(os.path.join(log_dir, f"loss_curve_{typo}_{target_word}.png"))
                        plt.clf()

                        # 保存 loss_buffer
                        with open(os.path.join(log_dir, f'loss_buffer_{typo}_{target_word}.pkl'), 'wb') as file:
                            pickle.dump(loss_buffer, file)
                            
                        # save adversarial image
                        x = denormalize(image_tensor).clone().cuda()
                        if zero_out:
                            x = x * inverse_mask
                        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
                        x_adv = x + adv_noise
                        x_adv = normalize(x_adv)
                        adv_img = denormalize(x_adv).detach().cpu()
                        adv_img = adv_img.squeeze(0)
                        save_image(adv_img, os.path.join(log_dir, f"adv_img_{typo}_{target_word}.png"))
                
                sys.stdout = sys.__stdout__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()
    main(args)
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Execution time: {int(execution_time)} minutes")