import argparse
import glob
import imp
import os
from tqdm import tqdm
import sys
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import TextStreamer

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

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

    log_root = 'logs'
    data_root = 'dataset/classification'
    total_datasets = [['species-r1'],]
    
    # 提问
    prefixs =  ["Describe the image first, then answer with the option's letter from the given choices.3 "]
    questions = ['What entity is depicted in the image? (a) {} (b) {}']
    
    for q in questions:
        for p in prefixs:
            question = p + q
            for datasets in total_datasets:
                for dataset in datasets:
                    # 重定向输出到文件
                    with open(os.path.join(log_root, dataset + '-' + model_name + '-' + question), 'w') as f:
                        sys.stdout = f

                        image_files = []
                        for extension in ["*.jpg", "*.jpeg", "*.png"]:
                            image_files.extend(glob.glob(os.path.join(data_root, dataset, extension)))

                        for image_file in tqdm(image_files):
                            image = load_image(image_file)
                            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

                            conv = conv_templates[args.conv_mode].copy()
                            if "mpt" in model_name.lower():
                                roles = ('user', 'assistant')
                            else:
                                roles = conv.roles

                            count = 0
                            outputs = None
                            print(image_file)
                            while True:
                                label = image_file.split('.')[0].split('-')[-2]
                                mislabel = image_file.split('.')[0].split('-')[-1]
                                if count==0:
                                    # inp = "Please describe the contents of this image without considering any typographical errors that might be present."
                                    inp = "Please describe the image without considering any text that may be printed on it. Focus on the visual elements and the overall scene, ignoring any written information."
                                if count==1:
                                    conv = conv_templates[args.conv_mode].copy()
                                    inp = ("Answer with the option's letter from the given choices directly. " + 'What entity is depicted in the image? (a) {} (b) {}').format(label, mislabel)
                                    inp = outputs + '\n' + inp
                                    
                                print(inp)
                                print(f"{roles[1]}: ", end="")

                                if image is not None:
                                    # first message
                                    if model.config.mm_use_im_start_end:
                                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                                            DEFAULT_IM_END_TOKEN + '\n' + inp
                                    else:
                                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                                    conv.append_message(conv.roles[0], inp)
                                    image = None
                                else:
                                    # later messages
                                    conv.append_message(conv.roles[0], inp)
                                conv.append_message(conv.roles[1], None)
                                prompt = conv.get_prompt()

                                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                                keywords = [stop_str]
                                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                                with torch.inference_mode():
                                    output_ids = model.generate(
                                        input_ids,
                                        images=image_tensor,
                                        do_sample=True,
                                        temperature=args.temperature,
                                        max_new_tokens=1024,
                                        streamer=streamer,
                                        use_cache=True,
                                        stopping_criteria=[stopping_criteria])

                                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                                conv.messages[-1][-1] = outputs

                                count += 1
                                if count == 2:
                                    print()
                                    break

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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = int((end_time - start_time) / 60)
    print(f"Elapsed Time: {elapsed_time} minutes")