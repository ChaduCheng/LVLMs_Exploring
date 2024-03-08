# -*- coding: utf-8 -*-
"""CLIP GradCAM Visualization

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/109CyyynwuPqgwVx_RTF_QEvwQEVaxD0N

# CLIP GradCAM Colab

This Colab notebook uses [GradCAM](https://arxiv.org/abs/1610.02391) on OpenAI's [CLIP](https://openai.com/blog/clip/) model to produce a heatmap highlighting which regions in an image activate the most to a given caption.

**Note:** Currently only works with the ResNet variants of CLIP. ViT support coming soon.
"""

# !pip install ftfy regex tqdm matplotlib opencv-python scipy scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install git+https://github.com/openai/CLIP.git -i https://pypi.tuna.tsinghua.edu.cn/simple

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur, save_path):
    dpi = 300  # 设置分辨率为300 DPI
    figsize = (10, 10)  # 调整图像大小，这里设为10x10英寸
    plt.figure(figsize=figsize, dpi=dpi)
    att_map_img = getAttMap(img, attn_map, blur)  # 获取关注度映射图像
    plt.imshow(att_map_img)
    plt.axis("off")  # 移除坐标轴
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')  # 保存图像
    plt.show()  # 显示图像

def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

# GradCAM: Gradient-weighted Class Activation Mapping
# Our gradCAM implementation registers a forward hook on the model at the specified layer. This allows us to save the intermediate activations and gradients at that layer.
# To visualize which parts of the image activate for a given caption, we use the caption as the target label and backprop through the network using the image as the input.
# In the case of CLIP models with resnet encoders, we save the activation and gradients at the layer before the attention pool, i.e., layer4.

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


clip_model = "RN50" #["RN50", "RN101", "RN50x4", "RN50x16"]
saliency_layer = "layer4" #["layer4", "layer3", "layer2", "layer1"]
blur = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device, jit=False)

# 检查文件夹是否存在
gradcam_folder = 'clip_gradcam'
if os.path.exists(gradcam_folder):
    shutil.rmtree(gradcam_folder)
    os.makedirs(gradcam_folder)
else:
    os.makedirs(gradcam_folder)

# image_folder = 'images/species-r0'
# image_folder = 'images/species-r1'
image_folder = 'images/counting-r0'
# image_folder = 'images/counting-r1'

for image_path in tqdm(os.listdir(image_folder)):
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_caption_list = [image_path.split('.')[0].split('-')[-1], image_path.split('.')[0].split('-')[-2]]
        for image_caption in image_caption_list:

            image_input = preprocess(Image.open(image_folder+'/'+image_path)).unsqueeze(0).to(device)
            image_np = load_image(image_folder+'/'+image_path, model.visual.input_resolution)
            text_input = clip.tokenize([image_caption]).to(device)

            attn_map = gradCAM(
                model.visual,
                image_input,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()

            save_path = gradcam_folder+'/'+image_path.split('.')[0]+'_'+image_caption+'.jpg'
            viz_attn(image_np, attn_map, blur, save_path)