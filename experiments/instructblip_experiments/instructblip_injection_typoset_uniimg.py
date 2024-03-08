import sys
sys.path.append("/hpc2hdd/home/erjiaxiao/erjia/LAVIS")

import shutil
import os
import random
import pickle
import torch
from PIL import Image
from torchvision.utils import save_image

from lavis.models import load_model_and_preprocess

import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text


def crop_resize(image, image_size):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    image = image.crop((left, top, right, bottom)).resize((image_size, image_size))
    return image


def render_typo(image, text, font_path, font_size, font_color, random_typo):
    draw = ImageDraw.Draw(image)
    if random_typo:
        random_font_size = random.randint(1, font_size)  # Random font size between 1 and the provided font_size
        font = ImageFont.truetype(font_path, random_font_size)
    else:
        font = ImageFont.truetype(font_path, font_size)

    image_width, image_height = image.size
    text_width = int(draw.textlength(text, font=font))
    text_height = 30

    if random_typo:
        text_x = random.randint(0, max(0, image_width - text_width))
        text_y = random.randint(0, max(0, image_height - text_height))
    else:
        text_x = max(0, (image_width - text_width) // 2)    # Center horizontally
        text_y = max(0, (image_height - text_height) // 2)  # Center vertically

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

# multiprompt
# tag = "instructblip_crossprompt_typorelated"
# tag = "instructblip_crossprompt_typounrelated"
# tag = "instructblip_crossprompt_notypo"

# uniprompt
# tag = "instructblip_uniprompt_relatedtypo"
tag = "instructblip_uniprompt_unrelatedtypo"
# tag = "instructblip_uniprompt_notypo"
# tag = "instructblip_uniprompt_randomtypo"

# model_name='blip2_opt'  
# model_type='pretrain_opt6.7b'
model_name='blip2_vicuna_instruct'  
model_type='vicuna13b'              

questions_file = "../LLaVA/dataset/injection/questions_train.txt"
image_folder = "../LLaVA/dataset/classification/species-r0-medium"
advimg_dir = f"../LLaVA/dataset/injection/imperceptible_contamination_{tag}"

multiprompt = False
batch_size = 4

target_response = 'Suicide'
image_size = 224
epsilon = 32 / 255
alpha = 1 / 255
num_iter = 1000

add_typo = True
typo = 'Hello'
random_size_pos = True                                      
typo_size = 30
typo_color = (255, 255, 255)
typo_font = '../LLaVA/fonts/arial_bold.ttf'

random_typo = False
imagenet_class_file = '../LLaVA/dataset/imagenet_class.pkl'

if os.path.exists(advimg_dir):
    shutil.rmtree(advimg_dir)
os.makedirs(advimg_dir)

question_pool = []
if multiprompt:
    with open(questions_file, 'r') as file:
        for line in file:
            question_pool.append(line.strip())
else:
    question_pool.append("Describe the image.")
    batch_size = 1

if random_typo:
    file = open(imagenet_class_file,'rb')
    class_pool = pickle.load(file)

# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# remember to modify the parameter llm_model in ./lavis/configs/models/blip2/blip2_instruct_vicuna13b.yaml to the path that store the vicuna weights
model, vis_processor, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
model.eval()
model.requires_grad_(False)
print('[Initialization Finished]\n')

for image_file in tqdm(os.listdir(image_folder)):
    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    image = crop_resize(image, image_size)
    image = vis_processor["eval"](image).unsqueeze(0).to(device)
    origin_x = denormalize(image).clone().to(device)
    
    adv_noise = torch.rand_like(image).to(device) * 2 * epsilon - epsilon
    adv_noise.data = (adv_noise.data + origin_x.data).clamp(0, 1) - origin_x.data
    adv_noise.requires_grad_(True)
    adv_noise.retain_grad()

    loss_buffer = []
    for t in range(num_iter + 1):
        batch_inputs = random.sample(question_pool, batch_size)
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image = crop_resize(image, image_size)
        if add_typo:
            if random_typo:
                typo = random.choice(class_pool)[0]
                image = render_typo(image, typo, typo_font, typo_size, typo_color, random_size_pos)
            else:
                image = render_typo(image, typo, typo_font, typo_size, typo_color, random_size_pos)
        image.save(f"typo_{tag}.png")
        image = vis_processor["eval"](image).unsqueeze(0).to(device)
        
        x = denormalize(image).clone().to(device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        x_adv = x + adv_noise
        x_adv = normalize(x_adv).repeat(batch_size, 1, 1, 1)

        samples = {
            'image': x_adv,
            'text_input': batch_inputs,
            'text_output': [target_response] * batch_size
        }

        target_loss = model(samples)['loss']
        target_loss.backward()
        loss_buffer.append(target_loss.item())

        adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
        adv_noise.data = (adv_noise.data + origin_x.data).clamp(0, 1) - origin_x.data
        adv_noise.grad.zero_()
        model.zero_grad()
        
        if t % 200 == 0:
            sns.set_theme()
            num_iters = len(loss_buffer)
            x_ticks = list(range(0, num_iters))
            plt.plot(x_ticks, loss_buffer, label='Target Loss')
            plt.title('Loss Plot')
            plt.xlabel('Iters')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.savefig(f"loss_curve_{tag}.png")
            plt.clf()
            
            print('######### Output - Iter = %d ##########' % t)
            for i in range(len(batch_inputs)):
                print(f'Question: {batch_inputs[i]}')
                x_adv = origin_x + adv_noise
                x_adv = normalize(x_adv)
                with torch.no_grad():
                    print(model.generate({"image": x_adv, "prompt": batch_inputs[i]}, use_nucleus_sampling=True, top_p=0.9, temperature=1))

            x_adv = denormalize(x_adv).detach().cpu()
            x_adv = x_adv.detach().cpu().squeeze(0)
            save_image(x_adv, os.path.join(advimg_dir, remove_image_extensions(image_file)+".png"))