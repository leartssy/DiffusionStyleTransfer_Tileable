import random
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import EXAMPLE_DOC_STRING
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import load_image
from diffusers.utils.doc_utils import replace_example_docstring

import numpy as np
import torch
import glob
from typing import List, Optional, Union
import PIL.Image
import os
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from pnp_utils_style import *
from pnp_utils_style import register_time
import time
#TexTile Integration
import textile
from textile.utils.image_utils import read_and_process_image
import torch.nn.functional as F

def load_img1(self, image_path, pro_size=512, keep_aspect_ratio=True):
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    if keep_aspect_ratio:
        if orig_w >= orig_h:
            target_w, target_h = pro_size, int(orig_h * (pro_size / orig_w))
        else:
            target_h, target_w = pro_size, int(orig_w * (pro_size / orig_h))
    
        target_w, target_h = (target_w // 8) * 8, (target_h // 8) * 8
        image_pil = T.Resize((target_h, target_w), interpolation=T.InterpolationMode.LANCZOS)(image_pil)
    else:
        image_pil = T.Resize(pro_size, interpolation=T.InterpolationMode.LANCZOS)(image_pil)
        image_pil = T.CenterCrop(pro_size)(image_pil)
        
    return image_pil


class PNP(nn.Module):
    def __init__(self, pipe, config, textile_guidance_scale, pnp_attn_t=20, pnp_f_t=20): #textile guidance_scale for texTile
        super().__init__()
        self.config = config
        self.device = config.device
        #old code without TexTile:use to switch without TexTile
        #self.pipe = pipe
        
        #use custom Blip class instead
        self.pipe = BLIP_With_Textile(pipe, textile_guidance_scale, config.alpha, config.ddim_steps)
        
        self.pnp_attn_t = pnp_attn_t
        self.pnp_f_t = pnp_f_t
        #end custom class

        self.pipe.scheduler.set_timesteps(config.ddim_steps, device=self.device)

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.pipe.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.pipe.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self.pipe, self.qk_injection_timesteps, weight=self.config.attention_weight)
        register_conv_control_efficient(self.pipe, self.conv_injection_timesteps)
        return self.qk_injection_timesteps
    

    def run_pnp(self, content_latents, style_latents, style_file, content_fn="content", style_fn="style"):
        
        all_times = []
        
        pnp_f_t = int(self.config.ddim_steps * self.config.alpha)
        pnp_attn_t = int(self.config.ddim_steps * self.config.alpha)
        

        content_step = self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        cond_subject = ""
        tgt_subject = ""
        text_prompt_input = ""
        #dynamic latent size
        lat_h, lat_w = content_latents.shape[-2:]
        current_height = lat_h * 8
        current_width = lat_w * 8
        cond_image = load_img1(self,style_file, pro_size=self.config.pro_size, 
                               keep_aspect_ratio=self.config.keep_aspect_ratio)
        guidance_scale = self.config.guidance_scale #previously 7.5
        textile_guidance = self.config.textile_guidance_scale
        is_tileable = self.config.is_tileable
        #previous code:num_inference_steps = 50
        num_inference_steps = self.config.ddim_steps
        negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
        
        init_latents = content_latents[-1].unsqueeze(0).to(self.device)

        

        output = self.pipe(
            content_latents,
            style_latents,
            text_prompt_input,
            cond_image,
            cond_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            latents=init_latents,
            height=current_height,
            width=current_width,
            content_step=content_step,
        ).images

        
        #output[0].save(f'{self.config.output_dir}/{self.config.prefix_name}+{os.path.basename(content_fn)}+{os.path.basename(style_fn)}.png')

        return output
        

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
class BLIP(BlipDiffusionPipeline):    
    @torch.no_grad()
    def __call__(
        self,
        content_latents,
        style_latents,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        content_step = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        device = self._execution_device

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        batch_size = len(prompt)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)

        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            register_time(self, t.item())
            do_classifier_free_guidance = guidance_scale > 1.0
            
            if t in content_step:
                content_lat = content_latents[t].unsqueeze(0)
                latent_model_input = torch.cat([content_lat] + [latents] * 2 ) if do_classifier_free_guidance else latents
            else:
                style_lat = style_latents[t].unsqueeze(0)
                latent_model_input = torch.cat([style_lat] + [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.tensor(latent_model_input, dtype=torch.float16)

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]
            
        latents = (latents).half()
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

#Create a new custom class that integrates TexTile
class BLIP_With_Textile(BlipDiffusionPipeline):    
    #different code:
    # Custom constructor to accept and initialize the TexTile metric
    def __init__(self, original_pipe, textile_guidance_scale, alpha, ddim_steps):
        # 2. Store new parameters and initialize the TexTile loss
        self._textile_guidance_scale = textile_guidance_scale
        self._original_pipe = original_pipe
        self._alpha = alpha
        self._ddim_steps = ddim_steps
        if self._textile_guidance_scale > 0:
            # Initialize the TexTile loss function (the metric)
            print(f"[INFO] Initializing TexTile metric with scale: {self._textile_guidance_scale}")
            self.textile_metric = textile.Textile().to(original_pipe.device) 
        else:
            self.textile_metric = None
    #end of different code
        
        # 1. Copy all components from the original loaded pipeline (e.g., vae, unet, scheduler)
        components_dict = original_pipe.components
        super().__init__(**components_dict)
    @torch.no_grad()
    def __call__(
        self,
        content_latents,
        style_latents,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        content_step = None,
        height: int = 512, #passed dynamically from run_pnp
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        device = self.unet.device

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        batch_size = len(prompt)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)

        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        #calculate where style injection is expected to start
        style_start_index = int(num_inference_steps * self._alpha)
        style_stop_index = int(num_inference_steps)
        #textile start -> delay running of textile into last steps
        tex_start = 0.1
        Textile_start_step = int(num_inference_steps * tex_start)
        

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            register_time(self, t.item())
            do_classifier_free_guidance = guidance_scale > 1.0
            
            #safety if style and content latents not same aspect ratio
            target_h, target_w = latents.shape[-2:] #size of current

            if t in content_step:
                content_lat = content_latents[i].unsqueeze(0)
                latent_model_input = torch.cat([content_lat] + [latents] * 2 ) if do_classifier_free_guidance else latents
            elif i < style_stop_index:
                style_lat = style_latents[i].unsqueeze(0)
                
                #if style latents aspect ratio doesn´t match
                if style_lat.shape[-2:] != (target_h, target_w):
                    style_lat = F.interpolate(
                        style_lat,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                latent_model_input = torch.cat([style_lat] + [latents] * 2) if do_classifier_free_guidance else latents
            
            else:
                latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents

            latent_model_input = latent_model_input.to(device).half()

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            ####Insert TextTile Guidance code
            #try integrating textile as ramp
            Ramp_start = Textile_start_step #starts at start percent
            Ramp_end = int(num_inference_steps)
            Max_scale = self._textile_guidance_scale
            Textile_skip = 20

            current_textile_scale = 0.0
            if i>= Ramp_start:
                  ramp_progress = (i-Ramp_start) / (Ramp_end - Ramp_start)
                  ramp_progress = min(1.0,ramp_progress)
                  current_textile_scale = Max_scale * ramp_progress

            is_textile_step = (i % Textile_skip == 0) and not (i == num_inference_steps)
            if self.textile_metric is not None and self._textile_guidance_scale > 0 and is_textile_step:
              #Only activate TexTile during the style-focused steps ---
                if i >= Textile_start_step:
                  # 1. Decode the current latents to pixel space (required by TexTile)
                  # Note: This decoding step is computationally expensive and needs to be done carefully.
                  # We temporarily enable gradients and clone latents for safety
                  print(f"[TexTile Debug] Step {i}: TexTile ACTIVE (Late-Stage Correction)")
                  latents_clone = latents.clone().detach().to(torch.float16).requires_grad_(True)
        
                  # Set latents to require grad for backpropagation
                  
                  with torch.enable_grad(): # Ensure gradients are enabled for the tileability loss

                      # Temporarily decode to get the image for the metric  
                      current_image = self.vae.decode(latents_clone / self.vae.config.scaling_factor, return_dict=False)[0]
                      
                      # 2. Calculate the differentiable TexTile metric
                      tileability_value = self.textile_metric(current_image) 
                      
                      # 3. Calculate the gradient of the metric
                      grad = torch.autograd.grad(tileability_value, latents_clone)[0]
                      #for debugging: checks if Textile gradient high enough to have influence
                      # Scale by the noise schedule's sigma to keep magnitude consistent
                      alpha_cumprod = self.scheduler.alphas_cumprod[i]
                      sigma = torch.sqrt(1 - alpha_cumprod)
                      #test:make scale bigger
                      grad_scaled = grad
                      # 4. Apply the guidance as an additional noise prediction component
                      noise_pred = noise_pred + current_textile_scale * grad_scaled.to(noise_pred.dtype) * sigma
                      #for debugging
                      if i % 5 == 0 or i == self.scheduler.num_inference_steps: 
                        print(f"[TexTile Debug] Step {i}/{self.scheduler.num_inference_steps-1} | Loss: {tileability_value.item():.5f} | Grad Norm: {grad.norm().item():.5f}")
                #for debugging: see when Textile inactive
                elif i % 10 == 0 or i == 0:
                    print(f"[TexTile Debug] Step {i}: TexTile INACTIVE (Content Injection Phase)")
              
            #### End of TexTile integration

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]
            
        latents = (latents).half()
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)









