import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torchvision.utils import save_image
from transformers import CLIPProcessor, CLIPModel


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

# model_name = "openai/clip-vit-large-patch14-336"
model_name = "openai/clip-vit-large-patch14"

if model_name == "openai/clip-vit-large-patch14":
    save_dir = os.path.join(save_dir, task+f'-224px-onetypo-esp{int(eps*255)}')
elif model_name == "openai/clip-vit-large-patch14-336":
    save_dir = os.path.join(save_dir, task+f'-336px-onetypo-esp{int(eps*255)}')
    
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

image_list = os.listdir(os.path.join(data_root, task+'-r0'))

for image_file in tqdm(image_list):
    loss_buffer = []
    
    init_img = Image.open(os.path.join(data_root, task+'-r0', image_file)).convert('RGB')
    target_img = Image.open(os.path.join(data_root, task+'-r1', image_file)).convert('RGB')
    init_inputs = processor(images=init_img, return_tensors="pt").to(device)
    target_inputs = processor(images=target_img, return_tensors="pt").to(device)

    adv_noise = torch.rand_like(init_inputs['pixel_values']).to(device)  * 2 * eps - eps
    x = denormalize(init_inputs['pixel_values']).clone().to(device)
    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
    adv_noise = adv_noise.to(device)
    adv_noise.requires_grad_(True)
    adv_noise.retain_grad()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    target_embedding = model.get_image_features(**target_inputs)

    for t in range(num_iter + 1):
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
            plt.savefig(os.path.join(log_dir, f"loss_curve_{task}_esp{int(eps*255)}.png"))
            plt.clf()
                
            # save adversarial image
            x_adv = x + adv_noise
            adv_img = x_adv.detach().cpu().squeeze(0)
            save_image(adv_img, os.path.join(save_dir, f"{remove_image_extensions(image_file)}.png"))