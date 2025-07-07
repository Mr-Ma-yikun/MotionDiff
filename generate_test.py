import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import glob  
from PIL import Image
import torch
from torchvision import utils
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F  

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_gird import DDIMSamplerWithGrad
from losses import FlowLoss
import re

import cv2  
import numpy as np  
  
def dilate_object(image, kernel_size=3, max_scale=1.12):  
    image = cv2.imread(image)

    # Convert image to grayscale for simplicity  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
      
    # Create a mask where background (255) is 0 and non-background is 255  
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)  

    # Find contours of the objects  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
      
    # Create a result image to draw the dilated objects  
    result = np.zeros_like(image) + 255
      
    # Iterate over each contour  
    for contour in contours:  
        # Get bounding box of the contour  
        x, y, w, h = cv2.boundingRect(contour)  
          
        # Calculate the scale factor for dilation  
        # We will use a simple heuristic: scale the object up to max_scale times its original size  
        # but ensure it doesn't exceed the image boundaries  
        new_w = int(min(w * max_scale, image.shape[1] - x))  
        new_h = int(min(h * max_scale, image.shape[0] - y))  

        # Get the region of interest (ROI) from the original image  
        roi = image[y:y+h, x:x+w]  
          
        # Calculate the position of the dilated object in the result image  
        result_y = y  
        result_x = x  
          
        # If the dilated object would go out of bounds, adjust the position accordingly  
        if new_w > w:  
            result_x -= (new_w - w) // 2  
            if result_x < 0:  
                result_x = 0  
                new_w = min(new_w, image.shape[1] - x)  
        if new_h > h:  
            result_y -= (new_h - h) // 2  
            if result_y < 0:  
                result_y = 0  
                new_h = min(new_h, image.shape[0] - y)  
          
        # Ensure the ROI for the result is within bounds  
        result_roi = result[result_y:result_y+new_h, result_x:result_x+new_w]  
          
        # Resize the original ROI to the new size and place it in the result image  
        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  
        result_roi[:resized_roi.shape[0], :resized_roi.shape[1]] = resized_roi  
      
    return result  


device = torch.device('cuda:0')
device2 = torch.device('cuda:1')

#-----------------------------------------------------------使用SD的推理ppl
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, logging

pipe = StableDiffusionPipeline.from_pretrained("/home/oseasy/mayikun/runwayml/stable-diffusion-v1-5/", 
                                               use_auth_token=True).to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device).eval()

def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    return model

def find_folders_with_prefix(root_dir, prefix):  
    folders = []  
    for root, dirs, files in os.walk(root_dir):  
        for dir_name in dirs:  
            if dir_name.startswith(prefix) and dir_name[len(prefix):].isdigit():  
                folders.append(os.path.join(root, dir_name))  
    return folders

def extract_imgs(s):  
    return int(s.rsplit('.', 1)[0].rsplit('/', 1)[-1])
  
def extract_latents(s):  
    return int(s.split('_')[-1])

def extract_flows(s):  
    parts = s.split('/')
    filename = parts[-1]  # 获取文件名部分，例如 'flow480.pth'  
    number_str = filename.split('flow')[1].split('.pth')[0]  # 提取序号部分，例如 '480'  
    return int(number_str)  # 将序号转换为整数

def extract_masks(s):  
    parts = s.split('/')
    filename = parts[-1]  # 获取文件名部分，例如 'flow480.pth'  
    number_str = filename.split('mask')[1].split('.pth')[0]  # 提取序号部分，例如 '480'  
    return int(number_str)  # 将序号转换为整数

def main():

    #-----------------------------------------------加载参数
    parser = argparse.ArgumentParser()
    # Generation args
    parser.add_argument("--save_dir", required=True, help='Path to save results')
    parser.add_argument("--batch_size_num", required=True, help='batch de shu liang')

    parser.add_argument("--num_samples", default=1, type=int, help='Number of samples to generate')
    parser.add_argument("--input_dir", type=str, required=True, help='location of src img, flows, etc.')
    parser.add_argument("--log_freq", type=int, default=0, help='frequency to log info')

    # Vanilla diffusion args
    parser.add_argument("--ddim_steps", type=int, default=500, help="number of ddim sampling steps. n.b. this is kind of hardcoded, so maybe don't change")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta, 0 => deterministic")
    parser.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="/home/oseasy/mayikun/runwayml/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt", help="path to checkpoint of model")
    parser.add_argument("--prompt", default='')

    # Guidance args
    parser.add_argument("--guidance_weight", default=300.0, type=float)
    parser.add_argument("--num_recursive_steps", default=10, type=int)
    parser.add_argument("--color_weight", default=120.0, type=float)
    parser.add_argument("--flow_weight", default=3.0, type=float)
    parser.add_argument("--clip_grad", type=float, default=200.0, help='amount to clip guidance gradient by. 0.0 means no clipping')

    opt = parser.parse_args()

    input_dir = Path(opt.input_dir)
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    torch.save(opt, save_dir / 'config.pth')

    #print(opt)

    # Load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}").to(device2)

    # Setup model
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.eval()

    guidance_energy = FlowLoss(opt.color_weight, opt.flow_weight).to(device)

    sampler = DDIMSamplerWithGrad(model, device, device2)

    torch.set_grad_enabled(False)

    # multi-view images
    target_images_path = input_dir
    image_files = glob.glob(os.path.join(target_images_path, '**', '*.jpg'), recursive=True)
    sorted_images = sorted(image_files, key=extract_imgs)  

    #print(sorted_images)

    src_imgs = []  

    for image_file in sorted_images:
        src_img = to_tensor(Image.open(image_file))[None] * 2 - 1
        src_img = src_img.to(device)
        src_imgs.append(src_img)  

    # -----------------------------------------------读取多视角光流

    flows_path = input_dir / 'flows' 
    flow_pth_files = glob.glob(f'{flows_path}/flow*.pth')  
    sorted_flows = sorted(flow_pth_files, key=extract_flows)  

    #print(sorted_flows)

    target_flows = []

    for flow_path in sorted_flows:
        target_flow = torch.load(flow_path).to(device)
        target_flows.append(target_flow)

    # multi-view masks

    masks_path = input_dir / 'flows' 
    mask_pth_files = glob.glob(f'{masks_path}/mask*.pth') 
    sorted_masks = sorted(mask_pth_files, key=extract_masks)  
 
    #print(sorted_masks)

    masks = []

    for mask_path in sorted_masks:
        mask = torch.load(mask_path).squeeze().to(device)
        masks.append(mask)

    # warps

    target_warp_path = input_dir / 'warp_imgs'
    warp_files = glob.glob(os.path.join(target_warp_path, '**', '*.png'), recursive=True)  
    sorted_warps = sorted(warp_files, key=extract_imgs)  

    #print(sorted_warps)
    warp_imgs = []
    warp_imgs_kuo =[ ]  

    warp_id = 0
    for warp_file in sorted_warps:
        warp_img = to_tensor(Image.open(warp_file))[None] *2 -1 
        warp_img = warp_img.to(device)
        warp_imgs.append(warp_img)

        warp_img_fangda_np = dilate_object(warp_file, kernel_size=3, max_scale=1.15)  

        cv2.imwrite(f'kuochong.{warp_id:05}.png', warp_img_fangda_np)
        
        warp_img_kuo = to_tensor(Image.open(f'kuochong.{warp_id:05}.png'))[None] *2 -1
        warp_img_kuo = warp_img_kuo.to(device)
        warp_imgs_kuo.append(warp_img_kuo) 
        
        warp_id = warp_id + 1 
    # -----------------------------------------------读取存储的latents

    #所有latents的文件夹，latents_1,2,3,...
    latent_paths = find_folders_with_prefix(input_dir, 'latents_') 
    sorted_latents = sorted(latent_paths, key=extract_latents)  
    #print(sorted_latents)
    all_latents = []  

    for latent_path in sorted_latents:

        latents = []

        for i in range(500):    
            latent = latent_path + '/' +  f'zt.{i:05}.pth'
            latents.append(torch.load(latent))
        
        cached_latent = torch.stack(latents)
        cached_latent = cached_latent.squeeze().to(device)
        all_latents.append(cached_latent)

    #(x, 500, 4, 64, 64)
    #[[1,2,..500], [1,2,..500], [1,2,..500]]
    cached_latents = torch.stack(all_latents, dim=0)

    #------------------------------加载prompt

    text_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        cond_embed = text_encoder(text_input.input_ids.to(device))[0]

    # And the uncond. input as before:
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embed = text_encoder(uncond_input.input_ids.to(device))[0]

    # Sample N examples
    for sample_index in range(opt.num_samples):

        sample_save_dir = save_dir / f'sample_{sample_index:03}'
        sample_save_dir.mkdir(exist_ok=True, parents=True)

        # Sample
        sample = sampler.sample(
                                            num_ddim_steps=opt.ddim_steps,
                                            cond_embed=cond_embed,
                                            uncond_embed=uncond_embed,
                                            batch_size=int(opt.batch_size_num),
                                            shape=[4, 64, 64],
                                            CFG_scale=opt.scale,
                                            eta=opt.ddim_eta,
                                            src_imgs=src_imgs,
                                            cached_latents=cached_latents,
                                            edit_masks=masks,
                                            num_recursive_steps=opt.num_recursive_steps,
                                            clip_grad=opt.clip_grad,
                                            guidance_weight=opt.guidance_weight,
                                            log_freq=opt.log_freq,
                                            results_folder=sample_save_dir,
                                            guidance_energy=guidance_energy,
                                            tar_flows = target_flows,
                                            warp_imgs=warp_imgs,
                                            device = device,
                                            device2 = device2,
                                            pipe = pipe,
                                            warp_imgs_kuo = warp_imgs_kuo, 

                                        )

        # Decode sampled latent
        sample_img = model.module.decode_first_stage(sample)
        sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)

        # Save useful unfo
        utils.save_image(sample_img, sample_save_dir / f'pred.png')

if __name__ == "__main__":
    main()
