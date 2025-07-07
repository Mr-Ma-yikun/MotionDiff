import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.ops as ops  
from torchvision import transforms  
from ldm.models.diffusion.grid_utils import make_grid, undo_grid

import torch.nn as nn

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

class DDIMSamplerWithGrad(object):
    def __init__(self, model, device, device2, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.device = device
        self.device2 = device2
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model.module.num_timesteps, verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               batch_size,
               shape,
               num_ddim_steps=500,
               src_imgs = None,
               cond_embed=None,
               uncond_embed=None,
               eta=0.,
               CFG_scale=1.,
               cached_latents=None,
               edit_masks=None,
               num_recursive_steps=1,
               clip_grad=0,
               guidance_weight=1.0,
               log_freq=0,
               results_folder=None,
               guidance_energy=None, 
               tar_flows = None,
               warp_imgs=None,
               device = None,  
               device2 = None, 
               pipe = None,
               warp_imgs_kuo = None

            ):
        
        # 存储图像和光流结果
        recon_save_dir = results_folder / 'recons'
        recon_save_dir.mkdir(exist_ok=True)
        flow_save_dir = results_folder / 'flow_viz'
        flow_save_dir.mkdir(exist_ok=True)

        # 加载Diffusion参数
        self.make_schedule(ddim_num_steps=num_ddim_steps, ddim_eta=eta, verbose=False)
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # Get shape and device info
        C, H, W = shape
        shape = (batch_size, C, H, W)
        b = batch_size

        # freeze
        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        warp_masks = torch.zeros(b, 3, 512, 512, device=src_imgs[0].device)

        for id_warp, warp_img in enumerate(warp_imgs):
            #mask=1, bg=0
            warp_mask = (warp_img != 1.0).to(torch.uint8)
            warp_masks[id_warp] = warp_mask

        # (b,4,64,64)
        noisy_latents = cached_latents[:, 499, :, :, :]

        warp_flag = 0

        # DDIM sampling loop
        for i, step in enumerate(iterator):

            index = total_steps - 1 - i
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            b, *_, device = *noisy_latents.shape, noisy_latents.device

            # Get variance schedule params
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            beta_t = a_t / a_prev

            if i >= 5:
                warp_flag = 1
                fusion_imgs = torch.zeros(b, 3, 512, 512, device=warp_imgs[0].device)  
                for warp_id, warp_img in enumerate(warp_imgs_kuo):
                    fusion_img = warp_img
                    fusion_imgs[warp_id] = fusion_img.squeeze(0)

                encoder_posterior = pipe.vae.to(device).eval().encode(fusion_imgs, return_dict=False)[0].sample().to(self.device)
                funsion_latents = encoder_posterior * 0.18215

                add_noise = noise_like(noisy_latents.shape, device, False)
                
                noisy_latents_object = a_t.sqrt() * funsion_latents + sqrt_one_minus_at * add_noise

            # For each step, do N extra "recursive" steps
            for j in range(num_recursive_steps):

                # (b, 4, 64, 64)
                gt_latents = cached_latents[:, 499-i, :, :, :]
                
                #对b张mask，更新latents, 替换背景Inversion
                for mask_id, edit_mask in enumerate(edit_masks):
                    # mask --> 4 64 64
                    noisy_latents[mask_id][edit_mask] = gt_latents[mask_id][edit_mask]

                # Set up grad (for differentiating guidance energy)
                torch.set_grad_enabled(True)
                #(b, 4, 64, 64)

                # gradient
                noisy_latent_grad = noisy_latents.detach().requires_grad_(True)
                
                x_in = torch.cat([noisy_latent_grad] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([uncond_embed, cond_embed] * b)
    
                # 8 4 64 64 ---  2 4 128 128
                grid_sample = make_grid(x_in)
                grid_prompt_embeds = c_in[:2]
                t_embeds = t_in[:2]

                noise_pred = pipe.unet.to(device).eval()(sample=grid_sample,timestep=t_embeds,encoder_hidden_states=grid_prompt_embeds,return_dict=False,)[0]
                
                e_t_uncond, e_t = undo_grid(noise_pred).chunk(2)

                e_t = e_t_uncond + CFG_scale * (e_t - e_t_uncond)

                #(b,4,64,64)
                pred_x0 = (noisy_latent_grad - sqrt_one_minus_at * e_t) / a_t.sqrt()
  
                recons_images = torch.zeros(b, 3, 512, 512, device=warp_imgs[0].device) 

                #with torch.no_grad():
                energy_total = torch.tensor([0.0], requires_grad=True).to(device)

                for inx, pre in enumerate(pred_x0):
                    pre_latent = 1 / 0.18215 * pre
                    recons_image = pipe.vae.to(device).eval().decode(pre_latent.unsqueeze(0), return_dict=False)[0]

                    #(1,2,512,512)
                    tar_flow = tar_flows[inx]

                    energy, info_loss = guidance_energy(recons_image, src_imgs[inx], tar_flow, device)
                    energy_total = energy_total + energy
                    recons_images[inx]=recons_image

                energy_ave = energy_total / b
                grad = torch.autograd.grad(energy_ave, noisy_latent_grad)[0]

                #调整梯度权重
                if warp_flag == 0:

                    grad = -grad * guidance_weight

                elif warp_flag ==1:

                    grad = -grad * guidance_weight

                # Clip gradient
                if clip_grad != 0:
                    grad_norm = torch.linalg.norm(sqrt_one_minus_at * grad.detach())
                    if grad_norm > clip_grad:
                        grad = grad / grad_norm * clip_grad

                # Update noise estimate with guidance gradiaent
                e_t = e_t - sqrt_one_minus_at * grad.detach()
                noisy_latent_grad = noisy_latent_grad.requires_grad_(False)

                # Save images
                #if i % log_freq == 0 and j == 0:
                # Save reconstruction
                temp = (recons_images + 1) * 0.5

                save_image(temp, recon_save_dir / f'xt.{i:05}.png')
                print(temp.size())
                info_loss['flow_im'].save(flow_save_dir / f'flow.{i:05}.png')

                del noisy_latent_grad, pred_x0, recons_images, grad, x_in

                torch.set_grad_enabled(False)

                # DDIM step
                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (noisy_latents - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

                    # random noise
                    noise = sigma_t * noise_like(noisy_latents.shape, device, False)

                    # DDIM step, get prev latent z_{t-1}
                    noisy_latent_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

                    # Inject noise (sample forward process) for recursive denoising
                    recur_noise = noise_like(noisy_latents.shape, device, False)
                    noisy_latents = beta_t.sqrt() * noisy_latent_prev + (1 - beta_t).sqrt() * recur_noise

                    del pred_x0, dir_xt, noise

            noisy_latents = noisy_latent_prev

            # fusion
            if warp_flag == 1:
                _,_,h,w = noisy_latents_object.shape
            
                # b 1 512 512 --> 
                warp_err = warp_masks.clone()
                #max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # 可调整 kernel_size 和 padding
                #max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
                #tensor_erode = -max_pool(-warp_err)
                
                # b 1 64 64
                mask = F.interpolate(warp_err[:, 0:1], (h,w))
                save_image(mask, recon_save_dir / f'mask.png')
				
                noisy_latents = mask * noisy_latents_object + (1 - mask) * noisy_latents

        return noisy_latents
