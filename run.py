from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline
import subprocess
import sys
from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

# suppress partial model loading warning
logging.set_verbosity_error()

import os
import os.path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils_style import *
import torchvision.transforms as T
from preprocess_style import get_timesteps, Preprocess
from pnp_style import PNP, BLIP, BLIP_With_Textile

#TexTileImports
import textile 
#endTextileImports

import torch.nn as nn

def make_model_circular(unet_model):
    """
    Patches a Stable Diffusion/BLIP-Diffusion U-Net to use circular padding.
    """
    count=0
    for name, module in unet_model.named_modules():
        # Find all 2D Convolutional layers
        if isinstance(module, nn.Conv2d):
            # Apply only to layers that actually use padding (usually kernel_size > 1)
            # 1x1 convolutions (kernel_size=1) typically don't need padding.
            if isinstance(module.padding, tuple):
                 if module.padding[0] > 0 or module.padding[1] > 0:
                     module.padding_mode = 'circular'
                     count +=1
            elif isinstance(module.padding, int):
                 if module.padding > 0:
                     module.padding_mode = 'circular'

    print(f"Circular padding enabled on U-Net on {count} layers.")
    return unet_model


def run(opt):
    
    model_key = Path(opt.model_key)
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
    
    #scheduler for Extrating Latents
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)

    #content_path = Path(opt.content_path)
    #content_path = [f for f in content_path.glob('*')]
    #style_path = Path(opt.style_path)
    #style_path = [f for f in style_path.glob('*')]

    content_path = get_file_list(opt.content_path)
    style_path = get_file_list(opt.style_path)

    is_tileable = opt.is_tileable
    out_size = opt.out_size
    gen_normal = opt.gen_normal
    alpha = opt.alpha
    
    extraction_path = "latents_reverse" if opt.extract_reverse else "latents_forward"
    base_save_path = os.path.join(opt.output_dir, extraction_path)
    os.makedirs(base_save_path, exist_ok=True)
    #use this for no Textile instead
    #pnp = PNP(blip_diffusion_pipe, opt)

    #TexTile pass guidance scale
    pnp = PNP(blip_diffusion_pipe, opt, opt.textile_guidance_scale)
    #

    # start of optimized latent loading code by gemini
    #2. Initialize Preprocess model ONCE before the main loop
    # This prevents redundant component loading for every image.
    model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
    print("[INFO] Preprocess model initialized once.")

    # Prepare timesteps once
    timesteps_to_save, _ = get_timesteps(
        scheduler, num_inference_steps=opt.ddpm_steps,
        strength=1.0,
        device=opt.device
    )
    if opt.steps_to_save < opt.ddpm_steps:
        timesteps_to_save = timesteps_to_save[-opt.steps_to_save:]


    all_content_latents = []
    print("\n[STEP 1] Loading/Extracting Content Latents...")
    for content_file in content_path:
        
        seed_everything(opt.seed)

        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(content_file))[0])
        os.makedirs(save_path, exist_ok=True)
        
        # Check for the *first* latent file to determine if extraction is needed
        check_path = os.path.join(save_path, 'noisy_latents_0.pt') 
        
        # Define a path for the *aggregated* latent file for faster loading
        aggregated_path = os.path.join(save_path, 'aggregated_latents.pt')

        content_latents = None
        
        # Try to load the single aggregated file first
        if os.path.exists(aggregated_path):
            print(f"Loading aggregated latents for {content_file}...")
            content_latents = torch.load(aggregated_path).to("cuda")
            
        elif not os.path.exists(check_path):
            # Extraction logic (if no files exist)
            print(f"No available latents, start extraction for {content_file}")
            _, content_latents = model.extract_latents(
                data_path=content_file,
                num_steps=opt.ddpm_steps,
                save_path=save_path,
                timesteps_to_save=timesteps_to_save,
                inversion_prompt=opt.inversion_prompt,
                extract_reverse=opt.extract_reverse
            )
            # You might want to save the content_latents aggregated here:
            torch.save(content_latents.cpu(), aggregated_path)

        else:
            # Fallback to the slow, step-by-step loading (if individual files exist but no aggregate)
            print(f"Loading individual latents for {content_file} (Slow I/O)...")
            content_latents_list = []
            for t in trange(opt.ddpm_steps, desc="Loading Content Steps"):
                latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                if os.path.exists(latents_path):
                    content_latents_list.append(torch.load(latents_path))
                else:
                    break # Stop if a file is missing
            
            if content_latents_list:
                content_latents = torch.cat(content_latents_list, dim=0).to("cuda")
            else:
                print(f"Warning: Could not load any latents for {content_file}.")
                continue # Skip this content file

        if content_latents is not None:
            all_content_latents.append(content_latents)


    all_style_latents = []
    print("\n[STEP 2] Loading/Extracting Style Latents...")

    num_style_steps = int(opt.ddpm_steps)
    
    for style_file in style_path:
        
        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(style_file))[0])
        os.makedirs(save_path, exist_ok=True)
        
        check_path = os.path.join(save_path, f'noisy_latents_0.pt')
        aggregated_path = os.path.join(save_path, 'aggregated_latents.pt')
        style_latents = None

        style_timesteps_to_save = timesteps_to_save[-num_style_steps:] if num_style_steps > 0 else []
        
        if os.path.exists(aggregated_path):
            print(f"Loading aggregated latents for {style_file}...")
            style_latents = torch.load(aggregated_path).to("cuda")

        elif not os.path.exists(check_path):
            print(f"No available latents, start extraction for {style_file}")
            model.scheduler.set_timesteps(opt.ddpm_steps)

            _, style_latents = model.extract_latents(
                data_path=style_file,
                num_steps=opt.ddpm_steps,
                save_path=save_path,
                timesteps_to_save=style_timesteps_to_save,
                inversion_prompt=opt.inversion_prompt,
                extract_reverse=opt.extract_reverse
            )
            # You might want to save the content_latents aggregated here:
            torch.save(style_latents.cpu(), aggregated_path)
        else:
            # Fallback to the slow, step-by-step loading
            print(f"Loading individual latents for {style_file} (Slow I/O)...")
            style_latents_list = []
            # Note: Style latents only load up to the alpha threshold 
            for t in trange(num_style_steps, desc="Loading Style Steps"):
                latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                if os.path.exists(latents_path):
                    style_latents_list.append(torch.load(latents_path))
                else:
                    break
            
            if style_latents_list:
                style_latents = torch.cat(style_latents_list, dim=0).to("cuda")
            else:
                print(f"Warning: Could not load any style latents for {style_file}.")
                continue

        if style_latents is not None:
            all_style_latents.append(style_latents[:num_style_steps])
            
    print("\n[STEP 3] Running PNP Style Transfer...")
    #set scheduler for generation phase
    blip_diffusion_pipe.scheduler.set_timesteps(opt.ddim_steps)
    newly_generated_paths = []
    if is_tileable:
        print("Enabling circular padding for tileability...")
        blip_diffusion_pipe.unet = make_model_circular(blip_diffusion_pipe.unet)
        pnp = PNP(blip_diffusion_pipe, opt, opt.textile_guidance_scale)
    
    base_res_path = []

    # The main execution loop is now ready to run without loading bottlenecks
    for content_latents, content_file in zip(all_content_latents, content_path):
        for style_latents, style_file in zip(all_style_latents, style_path):
            print(f"Transferring style from {style_file.name} to {content_file.name}")
            
            generated_images_list = pnp.run_pnp(content_latents, style_latents, style_file, content_fn=content_file, style_fn=style_file)
            generated_image_pil = generated_images_list[0]
            torch.cuda.empty_cache()
            #preserve alpha
            content_path_str = str(content_file)
            original_alpha = None
            try:
                with Image.open(content_path_str).convert("RGBA") as temp_img:
                    alpha_channel = temp_img.split()[-1]
                    if alpha_channel.getextrema() != (255,255):
                        original_alpha = temp_img.split()[-1].copy()
                        print(f"[SUCCESS] Alpha detected for {content_file.name}")
                    else:
                        print(f"[INFO] {content_file.name} is fully opaque")
            except Exception as e:
                print(f"[ERROR] Could not read alpha from {content_file.name}: {e}")

            content_fn_base = os.path.splitext(os.path.basename(content_file))[0]
            style_fn_base = os.path.splitext(os.path.basename(style_file))[0]
            output_size = out_size
            
            if is_tileable:
                
                #if color transfer off: correct the colors
                color_strength = opt.color_strength
                if color_strength < 1.0:

                    intensity = 1- color_strength
                    source_image = np.array(generated_image_pil.convert('RGB'))
                    print("Performing Color correction...")
                    final_im_blended = transfer_color(source_image,content_file,intensity)
                else:
                    source_image = np.array(generated_image_pil.convert('RGB'))
                    final_im_blended = source_image
                #normal blurry upscale
                #if target_dims != (final_im_blended.shape[1],final_im_blended.shape[0]):
                    #print(f"Upscaling to {output_size}px...")
                    #final_im_blended = cv2.resize(final_im_blended, target_dims,interpolation=cv2.INTER_LANCZOS4)
                generated_image_pil = Image.fromarray(final_im_blended)
                #Save the blended image
                out_fn = f'{opt.prefix_name}{content_fn_base}_s{style_fn_base}_tiled.png'
                save_path = os.path.join(opt.output_dir, out_fn)
                
                #don´t apply alpha because old doesn´t match tiled
                
                print("Saved witout Alpha")
                generated_image_pil.save(save_path)
                newly_generated_paths.append(save_path)
                print(f"Saved final blended image to {save_path}")

            else:
                
                out_fn = f'{opt.prefix_name}{content_fn_base}_s{style_fn_base}_raw.png'
                save_path = os.path.join(opt.output_dir, out_fn)
                
                #if color transfer off: correct the colors
                color_strength = opt.color_strength
                if color_strength < 1.0:

                    intensity = 1- color_strength
                    source_image = np.array(generated_image_pil.convert('RGB'))
                    print("Performing Color correction...")
                    generated_image_pil = transfer_color(source_image,content_file,intensity)
                    generated_image_pil = Image.fromarray(generated_image_pil)
                
                #apply alpha
                if original_alpha is not None:
                    print("Preserving Alpha...")
                    
                    alpha_resized = original_alpha.resize(generated_image_pil.size, Image.LANCZOS)
                    generated_image_pil.putalpha(alpha_resized)
                    
                generated_image_pil.save(save_path)
                                   
                newly_generated_paths.append(save_path)
                print(f"Saved raw generated image to {save_path}")
    
    #upscaling
    
    print(f"\n[CLEANUP]Cleanup for upscaling...")
    #clean up
    # Before loading Swin2SR
    del blip_diffusion_pipe
    del pnp
    if 'model' in locals(): del model

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() # Extra cleaning
    final_high_res_paths = []

    print(f"\n[STEP 2] AI Upscaling {len(newly_generated_paths)} images...")
    from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
    #load upscaler once
    model_id = "caidas/swin2sr-classical-sr-x4-64"
    processor = Swin2SRImageProcessor.from_pretrained(model_id)
    upscaler = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(opt.device).half()
    
    for img_path_str in newly_generated_paths:
        img_path = Path(img_path_str)   
        print(f"Upscaling (HF): {os.path.basename(img_path)}", flush=True)
        image = Image.open(img_path).convert("RGB")
        #option to keep aspect ratio
        orig_w, orig_h = image.size
        if opt.keep_aspect_ratio:
                # Scale the largest dimension to out_size
                if orig_w >= orig_h:
                    target_w = opt.out_size
                    target_h = int(orig_h * (opt.out_size / orig_w))
                else:
                    target_h = opt.out_size
                    target_w = int(orig_w * (opt.out_size / orig_h))
        else:
                target_w, target_h = opt.out_size, opt.out_size
        
        
        upscale_pass = 0
        
        #recursive upscaling with tiling
        while image.size[0] < target_w or image.size[1] < target_h:
            upscale_pass +=1
            # Add a 32px mirror buffer so the AI never sees a 'hard edge'
            pad = 32 
            img_np = np.array(image)
            padded_np = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode='wrap')
            # Note: Actually, for best results, use 'reflect' padding if using cv2, 
            # but for PIL, expanding by copying the edge is easiest:
            image = Image.fromarray(padded_np)
            curr_w, curr_h = image.size
            print(f"Upscale Pass {upscale_pass} (Using 16-tile grid)")
            
            # Check if we actually need another 4x pass
            # If we are already close to the target, we don't want a massive jump
            if curr_w >= target_w and curr_h >= target_h:
                break
            #split into 2x2 grid =4
            scale_factor = 4
            new_w, new_h = curr_w * scale_factor, curr_h * scale_factor
            stitched = Image.new("RGB", (new_w, new_h))
            grid_size = 4
            tile_w, tile_h = curr_w // grid_size, curr_h // grid_size
            overlap = 16


            for row in range(grid_size):
                for col in range(grid_size):
                    #define crop area with overlap
                    left = max(0, col * tile_w - overlap)
                    top = max(0,row * tile_h - overlap)
                    right = min(curr_w, (col + 1) * tile_w + overlap)
                    bottom = min(curr_h, (row + 1) * tile_h + overlap)

                    tile = image.crop((left,top,right,bottom))

                    # Prepare input
                    inputs = processor(tile, return_tensors="pt").to(opt.device).to(torch.float16)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = upscaler(**inputs)
        
                    # Post-process
                    output_tensor = outputs.reconstruction.data.squeeze().cpu().clamp(0, 1).numpy()
                    output_tensor = np.moveaxis(output_tensor, 0, -1)
                    upscaled_tile = Image.fromarray((output_tensor * 255).astype(np.uint8))

                    # How many pixels of overlap exist on the left/top of this specific tile?
                    crop_left = (col * tile_w - left) * scale_factor
                    crop_top = (row * tile_h - top) * scale_factor
                    
                    # The width/height we actually want to keep
                    keep_w = tile_w * scale_factor
                    keep_h = tile_h * scale_factor

                    clean_upscaled_tile = upscaled_tile.crop((
                        crop_left, 
                        crop_top, 
                        crop_left + keep_w, 
                        crop_top + keep_h
                    ))

                    # 4. Paste into the final canvas using 4x coordinates
                    paste_x = col * tile_w * scale_factor
                    paste_y = row * tile_h * scale_factor
                    #to hide seams only paste the non overlap center of tile, except for outer tiles

                    stitched.paste(clean_upscaled_tile, (paste_x, paste_y))
                    

                    #clean VRAM after every tile
                    del outputs, output_tensor
                    torch.cuda.empty_cache()
            # The pad also got upscaled by the scale_factor (4x)
            final_pad = pad * scale_factor

            # Crop the 'stitched' image to remove the buffer edges
            image = stitched.crop((
                final_pad, 
                final_pad, 
                stitched.width - final_pad, 
                stitched.height - final_pad
            ))

            if image.size[0] >= target_w and image.size[1] >= target_h:
                break
            if upscale_pass >= 2: break
        
        # scale to desired size
        upscaled_image = image
        # Swin2SR always outputs 4x. We resize its output to the user's specific target.
        if upscaled_image.size != (target_w, target_h):
            print(f"   > Adjusting size to {target_w}x{target_h}...")
            upscaled_image = upscaled_image.resize((target_w, target_h), resample=Image.LANCZOS)
        # reattach alpha
        with Image.open(img_path).convert("RGBA") as original_rgba:
            alpha = original_rgba.split()[-1]
            # Check if alpha is actually used (not just solid white)
            if alpha.getextrema() != (255, 255):
                print("   > Refining Alpha channel to match RGB sharpness...", flush=True)
                
                print("   > Fast resizing Alpha channel...")
                refined_alpha = alpha.resize((target_w, target_h), resample=Image.LANCZOS)
                
                # Optional: Sharpen the alpha to match AI sharpness
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Sharpness(refined_alpha)
                refined_alpha = enhancer.enhance(2.0) 
                
                upscaled_image.putalpha(refined_alpha)
            else:
                # If solid, just create a solid high-res alpha
                upscaled_image.putalpha(Image.new("L", (target_w, target_h), 255))
        if is_tileable:
            print("Blending Image Seams...")
            #integrate seam blending
            #convert PIL image to cv2 format
            im_np = np.array(upscaled_image.convert('RGB'))
            #determine gap size
            im_h, im_w , _ = im_np.shape
            #if gas is <1: treat as fraction of height, else as pixelwidth
            if opt.gap <1:
                gap_px = int(min(im_h, im_w) * opt.gap)
            else:
                gap_px = int(opt.gap)

            #store original image size
            im_origin_size = (im_w, im_h)
            #Apply the blending
            final_im_blended = apply_seam_blending(
                im_np,
                gap_px,
                opt.blurring,
                opt.min_ratio,
                im_origin_size=im_origin_size,
                maintain_size=opt.maintain_size
            )
            #convert back to pil
            final_im_blended = Image.fromarray(final_im_blended)
            upscaled_image = final_im_blended

        high_res_path = str(img_path)#.replace(".png", "_raw.png")
        upscaled_image.save(high_res_path)
        final_high_res_paths.append(high_res_path)
        
        print(f"[DONE] Saved: {os.path.basename(high_res_path)}")
    # Cleanup
    del upscaler, processor
    torch.cuda.empty_cache()

    if gen_normal:
        #print("Enabling Marigold Pipe for Normal Generation...")
        #from diffusers import MarigoldNormalsPipeline

        #marigold_pipe = MarigoldNormalsPipeline.from_pretrained(
            #"prs-eth/marigold-normals-v1-1",
           # variant="fp16",
           # torch_dtype=torch.float16
        #).to("cuda")

        strength = opt.normal_strength

        #normal map generation
        for img_path_str in final_high_res_paths:
            img_path = Path(img_path_str)
            input_img = Image.open(img_path).convert("RGB")

            #AI normals
            # final_normal = generate_normal(input_img, marigold_pipe,strength)
            #sobel normals
            final_normal = generate_sobel_normal(input_img, strength)
            #if is_tileable: #Also get rid of newly created seams in normal map
                #print("Blending Normal Seams...")
                #integrate seam blending
                #convert PIL image to cv2 format
                #im_np = np.array(final_normal.convert('RGB'))
                #determine gap size
                #im_h, im_w , _ = im_np.shape
                #if gas is <1: treat as fraction of height, else as pixelwidth
                #if opt.gap <1:
                    #gap_px = int(min(im_h, im_w) * opt.gap)
                #else:
                    #gap_px = int(opt.gap)

                #store original image size
                #im_origin_size = (im_w, im_h)
                #Apply the blending
                #final_normal = apply_seam_blending(
                    #im_np,
                    #gap_px,
                    #opt.blurring,
                    #opt.min_ratio,
                    #im_origin_size=im_origin_size,
                    #maintain_size=opt.maintain_size
                #)
                #final_normal = Image.fromarray(final_normal)
            #Save the normal map
            base_name = img_path.stem
            out_fn = f"{base_name}_normal.png"
            save_path = os.path.join(opt.output_dir, out_fn)
            final_normal.save(save_path)
            print(f"Saved final blended image to {save_path}")
        #del marigold_pipe
        #torch.cuda.empty_cache()
            
            
def transfer_color(source_image,target_image,intensity):
    
    target_image = load_img_file(str(target_image))
    cm = ColorMatcher()
    transferred_image = cm.transfer(src=source_image, ref=target_image, method='mkl')
    transferred_image = Normalizer(transferred_image).uint8_norm()
    #integrate intensity
    final_image = (intensity * transferred_image) + ((1.0 - intensity) * source_image)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return final_image

def blend_seams(image,gap,blur=3,min_ratio=0.2):
    #code adapted from: https://github.com/sagieppel/convert-image-into-seamless-tileable-texture/blob/main
    #tile the image first: cut the two opposite sides of image and subtract them pixel wise
    intensity_map = image.mean(2).astype(np.float32) #convert image to grayscale (=mean(2))
    difference_map = intensity_map[:gap] - intensity_map[-gap:] #calculate pixel-wise difference between top and botton band
    #if difference_map is close to 0: content in top and bottom band is similar, therefore good location for seam
    #balance the difference map: prevents only thin line when wanting to follow only one boundary
    while((difference_map>0).mean()< min_ratio): difference_map += 0.5
    while((difference_map<0).mean()< min_ratio): difference_map -= 0.5
    if ((difference_map>0).mean()< min_ratio): return image #smooth boundary: doesn´t need modification

    #turn difference map into binary map (decides which part is preferred at border)
    binary_map = (difference_map >0).astype(np.uint8) * 255
    trans_map = [binary_map]
    tmp_map = binary_map.copy()
    #dilate the binary map
    while (tmp_map.min() == 0):
        tmp_map = cv2.dilate(tmp_map, np.ones([3,3],np.uint8))
        trans_map.insert(0,tmp_map.copy())
    
    tmp_map = binary_map.copy()
    #erode binary map the more you get the other side of the merging zone the more empty map will be
    while(tmp_map.max()> 0):
        tmp_map = cv2.erode(tmp_map, np.ones([3, 3], np.uint8))
        trans_map.append(tmp_map.copy())

    # trans_maps=np.asarray(trans_maps)
    num_maps = len(trans_map)
    final_topology = []

    # Create transformation map that will be use to map the boundary the fully dilated in one side and fully erode in the other
    for i in range(gap):
        topology_indx = int(np.round(i * num_maps / gap))
        #   print(topology_indx,i)
        if topology_indx >= num_maps: topology_indx = num_maps - 1
        final_topology.append(trans_map[topology_indx][i])

    final_topology = np.array(final_topology).astype(np.float32)
    final_topology /= final_topology.max()
    if blur>-1:
       final_topology = cv2.blur(final_topology, [blur, blur])  # use gaussian blur to make the mixing softer
    
    # Use the transformation map to decide how the two sides of the image will intermingle
    gap_im = (1 - final_topology[:, :, None]) * image[:gap].astype(np.float32) + (final_topology)[:, :, None] * image[-gap:].astype(np.float32)
    gap_im = gap_im.astype(np.uint8)
    
    # use the new merge zone at the boundary of the image
    final_im = np.concatenate([gap_im[int(gap / 2):], image[gap:-gap], gap_im[:int(gap / 2)]], axis=0)
    
    return  final_im

def apply_seam_blending(image,gap_px,blur,min_ratio,im_origin_size=None,maintain_size=False):
    #vertical blending
    final_im = blend_seams(image,gap_px,blur, min_ratio)
    #horizontal blending: rotate 90degrees and tile again
    final_im = np.rot90(final_im)
    final_im = blend_seams(final_im,gap_px,blur,min_ratio)
    #rotate back
    final_im = cv2.rotate(final_im,cv2.ROTATE_90_CLOCKWISE)

    #maintain size if wanted
    if maintain_size and im_origin_size is not None:
        final_im = cv2.resize(final_im,im_origin_size)
    
    return final_im

def generate_normal(image, pipe,strength=2.0,detail_boost=0.5):
    from PIL import Image, ImageFilter
    #test:blur the image before normal pipeline: brushstrokes and stylized effects less a problem
    #image = image.filter(ImageFilter.GaussianBlur(radius=2)) #gaussian blur
   
    
    import cv2
    #convert to BGR as opencv uses this
    img_array = np.array(image)
    image_cv = cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR)
    #median filter
    image_cv = cv2.medianBlur(image_cv,5)
    #bilateral filter
    smoothed = cv2.bilateralFilter(image_cv,d=15,sigmaColor=75,sigmaSpace=75)
    
    #find the edges and sharpen them
    #covnert to greyscale
    gray_smoothed = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_smoothed,cv2.CV_32F)#laplacian to find rock edges
    laplacian = (laplacian - laplacian.min()) / (laplacian.max()-laplacian.min() + 1e-6)
    laplacian = (laplacian * 2.0) -1.0
    detail_map = laplacian    
    clean_image_pil = Image.fromarray(cv2.cvtColor(smoothed,cv2.COLOR_BGR2RGB))

    #load image
    output = pipe(clean_image_pil,output_type="pt") #output math vectors
    #for strength calculations
    normals = output.prediction
    #permute to bring into right order: (Batch,Height,Width,Channels)
    normals = normals.permute(0,2,3,1)
    
    #adjust normal strength
    #seperate: format: Batch,Height,Width,Channels
    #: -> take everything, 0:1 -> start at 0, stop at 1 -> take only first channel
    x_normals = normals[:,:,:,0:1]
    y_normals = normals[:,:,:,1:2]
    z_normals = normals[:,:,:,2:3]
    detail_map = torch.from_numpy(detail_map).to(normals.device).unsqueeze(0).unsqueeze(-1)


    #scale only x and y -> make surface look bumpier
    scaled_x_normals = x_normals * strength + detail_map * detail_boost
    scaled_y_normals = y_normals * strength + detail_map * detail_boost

    #reconstruct the normal map, normalize vectors **2 = ^2
    #Wurzel aus x^2 +y^2 +z^2 ist normal vector
    squared_magnitude = scaled_x_normals**2 + scaled_y_normals**2 + z_normals**2
    #wurzel davon
    magnitude = torch.sqrt(squared_magnitude)
    #diffusion models sometimes produce "dead pixels"->value 0, avoid dividing by 0
    #clamp(min=1e-6) -> smallest possible number tiny ~0.0000001
    magnitude = torch.clamp(magnitude, min=1e-6)

    #put RGB channels back where they belong dim=3, divide by magnitude to normalize
    adjusted_normals = torch.cat([scaled_x_normals, scaled_y_normals, z_normals],dim=3) / magnitude

    #convert to numpy
    image_numpy = adjusted_normals[0].cpu().numpy()
    #convert from [-1,1] to [0,1], so screen can display it
    image_numpy_scaled = (image_numpy + 1.0) / 2.0
    #convert to 8-bit [0,255] and create PIL image
    image_uint8 = (image_numpy_scaled *255).clip(0,255).astype(np.uint8)
    #convert to PIL image
    final_im = Image.fromarray(image_uint8)
    return final_im

def generate_sobel_normal(image, strength=2.0):
    # 1. Convert to grayscale and blur slightly to reduce noise
    gray = np.array(image.convert("L")).astype(np.float32)
    
    # Use BORDER_WRAP for the blur so the edges don't get dark
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_WRAP)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Crop back to original size after blurred wrap
    gray = gray[5:-5, 5:-5]

    # 2. Calculate Gradients with Border Wrap
    # We pad with wrap border, calculate Sobel, then crop
    gray_padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_WRAP)

    # 2. Calculate Gradients (Sobel)
    # ksize=3 or 5 is standard for textures
    dx = cv2.Sobel(gray_padded, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_padded, cv2.CV_32F, 0, 1, ksize=3)

    # Crop the gradients back to the original size
    # This removes the 1px border added by copyMakeBorder
    dx = dx[1:-1, 1:-1]
    dy = dy[1:-1, 1:-1]

    # 3. Construct the Normal Map
    # We invert dy because image coordinates (top-down) are opposite to OpenGL normals
    x = -dx * strength
    y = -dy * strength
    z = np.ones_like(gray) * 255.0

    # 4. Normalize the vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm

    # 5. Map from [-1, 1] to [0, 255]
    res = np.stack([x, y, z], axis=-1)
    res = ((res + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(res)

def upscale_image_ai(image_pil, prompt, device="cuda"):
    """Use Real-ESRGAN to upscale the image with sharp details."""
    from diffusers import StableDiffusionUpscalePipeline
    import torch

    # Handle Alpha Channel
    original_rgba = image_pil.convert("RGBA")
    alpha = original_rgba.split()[-1]
    image_rgb = image_pil.convert("RGB")
    
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    ).to(device)
    
    upscaled_image = upscaler(
        prompt=prompt,
        image=image_pil,
        guidance_scale=7.5,
        num_inference_steps=20
    ).images[0]

    # Restore Alpha
    # Resize original alpha to match new high-res dimensions
    upscaled_alpha = alpha.resize(upscaled_image.size, resample=Image.LANCZOS)
    upscaled_image = Image.merge("RGBA", (*upscaled_image.split(), upscaled_alpha))

    # Clean up immediately
    del upscaler
    torch.cuda.empty_cache()

    return upscaled_image

def get_file_list(path_input):
    #if input is single file, use single file, if path, use folder
    path = Path(path_input)
    if path.is_file():
        return[path]
    elif path.is_dir():
        return [f for f in path.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
    else:
        print(f"[WARNING] Path {path_input} not found.")
        return[]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true"):
        return True
    elif value.lower() in ("false"):
        return False
    else: raise argparse.ArgumentTypeError(f"Boolean expected, but got {value}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content_path', type=str,
                        default='images/content')
    parser.add_argument('--style_path', type=str,
                        default='images/style')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ddpm_steps', type=int, default=999)
    parser.add_argument('--steps_to_save', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    #not hardcoded model key
    parser.add_argument('--model_key', type=str, required=True, help='Path to the directory containing the pretrained model files (e.g., blipdiffusion folder).')
    #Textile
    parser.add_argument('--textile_guidance_scale', type=float, default=0.0, help="Strength of the TexTile loss for tileability constraint (0.0 to disable).")
    parser.add_argument('--is_tileable', type=str_to_bool, default=False, help="Set to true or False for circular padding")
    parser.add_argument('--gen_normal', type=str_to_bool, default=False, help="Set to true or False for normal map generation")
    parser.add_argument('--color_strength', type=float, default=1.0, help="Strength of the color of style image for transfer (0.0-1.0).")


    parser.add_argument('--normal_strength', type=float, default=2.0, help="Strength of the Normal Map.")


    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    parser.add_argument('--prefix_name', type=str, default='')
    
    #additional arguments for blending seams
    parser.add_argument('--gap',type=float, default=0.12, help="Size of border that will be used for blending in pixel size or in percentage if under 1")
    parser.add_argument('--min_ratio',type=float, default=0.01, help='Used to ensure balanced blending')
    parser.add_argument('--blurring',type=int, default=3,choices=range(1,10,2), help="Size of Gaussian blur to make merging softer.Use odd numbers only.")
    parser.add_argument('--maintain_size',type=bool, default=False,help="maintain images default size")
    parser.add_argument('--out_size',type=int, default=2048, help="Output image size")
    parser.add_argument('--keep_aspect_ratio',type=bool, default=True,help="Keep original aspect ratio of content image")

    

    opt = parser.parse_args()

    run(opt)

