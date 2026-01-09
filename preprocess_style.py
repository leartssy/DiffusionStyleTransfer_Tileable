from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import os, time
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils_style import *
import torchvision.transforms as T


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, blip_diffusion_pipe, device, scheduler=None, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.use_depth = False

        # Create model
        self.vae = blip_diffusion_pipe.vae
        self.unet = blip_diffusion_pipe.unet
        self.tokenizer = blip_diffusion_pipe.tokenizer
        self.text_encoder = blip_diffusion_pipe.text_encoder

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = blip_diffusion_pipe.scheduler
        print(f'[INFO] loaded stable diffusion!')

        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
      text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                  truncation=True, return_tensors='pt')
      
      # --- PATCH 1 --- Add return_dict=False
      text_embeddings = self.text_encoder(input_ids=text_input.input_ids.to(device), return_dict=False)[0]
      
      uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    return_tensors='pt')
      
      # --- PATCH 2 --- Add return_dict=False
      uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.to(device), return_dict=False)[0]
      
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
      return text_embeddings
    
    #old version
    #def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        #text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    #truncation=True, return_tensors='pt')
        #text_embeddings = self.text_encoder(input_ids=text_input.input_ids.to(device))[0]
        #uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      #return_tensors='pt')
       # uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.to(device))[0]
        #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        #return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path, pro_size=512, keep_aspect_ratio=True):
        #old:image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        #for security: force square size
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        
        #ensure that dimensions are multiples of 8 because vae needs that
        if keep_aspect_ratio:
            #calculate dimensions based on pro_size
            if orig_w >= orig_h:
                target_w = pro_size
                target_h = int(orig_h * (pro_size / orig_w))
            else:
                target_h = pro_size
                target_w = int(orig_w * (pro_size / orig_h))
            #make multiples of 8
            target_w = (target_w // 8) * 8
            target_h = (target_h // 8) * 8
            #resize
            resize = T.Resize((target_h, target_w), interpolation=T.InterpolationMode.LANCZOS)
            image_pil = resize(image_pil)
        else:
            target_w, target_h = pro_size, pro_size

            transform = T.Compose([
                T.Resize(pro_size, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop(pro_size)
            ])
            
            image_pil = transform(image_pil)

        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        print(f"[INFO] Image loaded with resolution: {target_w}x{target_h}")
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                                timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        all_latents = []
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):

                if t not in timesteps_to_save:
                    break
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                all_latents.append(latent)
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        all_latents = torch.cat(all_latents)
        return all_latents

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    
            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt='', extract_reverse=False, pro_size=512, keep_aspect_ratio=True):
        
        print('extract_reverse', extract_reverse)  # default to False
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path, pro_size=pro_size, keep_aspect_ratio=keep_aspect_ratio)
        latent = self.encode_imgs(image)

        inverted_latents = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        
        inverted_x = inverted_latents[-1].unsqueeze(0)
        latent_reconstruction = self.ddim_sample(inverted_x, cond, save_path, save_latents=extract_reverse,
                                                 timesteps_to_save=timesteps_to_save)
        
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        if not extract_reverse:
            return rgb_reconstruction, inverted_latents
        else:
            return rgb_reconstruction, latent_reconstruction



def run(opt):
    # timesteps to save
    model_key = Path(opt.model_key)
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
    toy_scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    data_paths = Path(opt.data_paths)
    data_paths = [f for f in data_paths.glob('*')]

    
    for data_path in data_paths:
        start_time = time.time()
        toy_scheduler.set_timesteps(opt.ddpm_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.ddpm_steps,
                                                            strength=1.0,
                                                            device=device)
        timesteps_to_save = timesteps_to_save[-opt.steps_to_save:]
        seed_everything(opt.seed)

        extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
        save_path = os.path.join(opt.save_dir + extraction_path_prefix, os.path.splitext(os.path.basename(data_path))[0])
        os.makedirs(save_path, exist_ok=True)

        model = Preprocess(blip_diffusion_pipe, device, sd_version=opt.sd_version, hf_key=None)

        recon_image = model.extract_latents(data_path=data_path,
                                            num_steps=opt.ddpm_steps,
                                            save_path=save_path,
                                            timesteps_to_save=timesteps_to_save,
                                            inversion_prompt=opt.inversion_prompt,
                                            extract_reverse=opt.extract_reverse)
        end_time = time.time()

        print(end_time - start_time)
        T.ToPILImage()(recon_image[0][0]).save(os.path.join(save_path, f'recon.jpg'))


if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str,
                        default='./images/style')
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ddpm_steps', type=int, default=999)
    parser.add_argument('--steps_to_save', type=int, default=300)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    opt = parser.parse_args()
    run(opt)
