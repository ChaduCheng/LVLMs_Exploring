import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random

import torch
from torch import nn
from torchvision.utils import save_image
from transformers import CLIPProcessor, CLIPModel


def crop_resize(image, image_size):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    image = image.crop((left, top, right, bottom)).resize((image_size, image_size))
    return image
    

def render_typo_randomly(image, text, font_path, font_size, font_color):
    draw = ImageDraw.Draw(image)

    random_font_size = random.randint(font_size//2, font_size)
    font = ImageFont.truetype(font_path, random_font_size)

    image_width, image_height = image.size
    text_width = int(draw.textlength(text, font=font))
    text_height = 10

    text_x = random.randint(0, max(0, image_width - text_width))
    text_y = random.randint(0, max(0, image_height - text_height))

    draw.text((text_x, text_y), text, fill=font_color, font=font)
    # image.save('test.png')
    return image


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text


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


task = 'species'
data_root = 'dataset/classification/'
save_dir = "dataset/classification/"
log_dir = 'logs'
eps = 16 / 255
num_iter = 500
alpha = 1 / 255

typo_size = 30
typo_color = (255, 255, 255)
typo_font = 'fonts/arial_bold.ttf'

model_name = "openai/clip-vit-large-patch14-336"
# model_name = "openai/clip-vit-large-patch14"

if model_name == "openai/clip-vit-large-patch14":
    img_size = 224
    save_dir = os.path.join(save_dir, task+f'-{img_size}px-typoset-esp{int(eps*255)}')
elif model_name == "openai/clip-vit-large-patch14-336":
    img_size = 336
    save_dir = os.path.join(save_dir, task+f'-{img_size}px-typoset-esp{int(eps*255)}')
    
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

image_list = os.listdir(os.path.join(data_root, task+'-r0'))

for image_file in tqdm(image_list):
    
    loss_buffer = []
    typo = remove_image_extensions(image_file).split('-')[-1]
    
    init_img = Image.open(os.path.join(data_root, task+'-r0', image_file)).convert('RGB')
    init_img = crop_resize(init_img, img_size)
    init_inputs = processor(images=init_img, return_tensors="pt").to(device)
    
    adv_noise = torch.rand_like(init_inputs['pixel_values']).to(device)  * 2 * eps - eps
    x = denormalize(init_inputs['pixel_values']).clone().to(device)
    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
    adv_noise = adv_noise.to(device)
    adv_noise.requires_grad_(True)
    adv_noise.retain_grad()
    
    for t in range(num_iter + 1):
        target_img = render_typo_randomly(init_img.copy(), typo, typo_font, typo_size, typo_color)
        target_inputs = processor(images=target_img, return_tensors="pt").to(device)
        
        x_adv = x + adv_noise
        x_adv = normalize(x_adv)
        init_inputs['pixel_values'] = x_adv
        init_embedding = model.get_image_features(**init_inputs)
        target_embedding = model.get_image_features(**target_inputs)
        
        cos_sim = cos(init_embedding, target_embedding)
        loss = 1 - cos_sim
        loss.backward()
        adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-eps, eps)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.grad.zero_()
        model.zero_grad()
        loss_buffer.append(loss.item())
        
        if t % 50 == 0:   
            # plot loss
            sns.set_theme()
            num_iters = len(loss_buffer)
            x_ticks = list(range(num_iters))
            plt.plot(x_ticks[:len(loss_buffer)], loss_buffer, label=f"epsilon {int(eps*255)}")
            plt.title('Loss Plot')
            plt.xlabel('Iters')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.savefig(os.path.join(log_dir, f"loss_curve_{task}_{img_size}px_esp{int(eps*255)}.png"))
            plt.clf()
                
            # save adversarial image
            x_adv = x + adv_noise
            adv_img = x_adv.detach().cpu().squeeze(0)
            save_image(adv_img, os.path.join(save_dir, f"{remove_image_extensions(image_file)}.png"))