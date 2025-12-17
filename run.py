from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline

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
    for name, module in unet_model.named_modules():
        # Find all 2D Convolutional layers
        if isinstance(module, nn.Conv2d):
            # Apply only to layers that actually use padding (usually kernel_size > 1)
            # 1x1 convolutions (kernel_size=1) typically don't need padding.
            if isinstance(module.padding, tuple):
                 if module.padding[0] > 0 or module.padding[1] > 0:
                     module.padding_mode = 'circular'
            elif isinstance(module.padding, int):
                 if module.padding > 0:
                     module.padding_mode = 'circular'

    print("Circular padding enabled on U-Net.")
    return unet_model


def run(opt):
    
    model_key = Path(opt.model_key)
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
    
    
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)
    content_path = Path(opt.content_path)
    content_path = [f for f in content_path.glob('*')]
    style_path = Path(opt.style_path)
    style_path = [f for f in style_path.glob('*')]
    is_tileable = opt.is_tileable
    gen_normal = opt.gen_normal
    
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
    for style_file in style_path:
        
        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(style_file))[0])
        os.makedirs(save_path, exist_ok=True)
        
        check_path = os.path.join(save_path, f'noisy_latents_0.pt')
        aggregated_path = os.path.join(save_path, 'aggregated_latents.pt')
        style_latents = None

        style_timesteps_to_save = timesteps_to_save[-int(opt.ddpm_steps*opt.alpha):]
        
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
            num_style_steps = int(opt.ddpm_steps * opt.alpha) 
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
            all_style_latents.append(style_latents)
            
    print("\n[STEP 3] Running PNP Style Transfer...")
    
    if is_tileable:
        print("Enabling circular padding for tileability...")
        blip_diffusion_pipe.unet = make_model_circular(blip_diffusion_pipe.unet)
        pnp = PNP(blip_diffusion_pipe, opt, opt.textile_guidance_scale)


    # The main execution loop is now ready to run without loading bottlenecks
    for content_latents, content_file in zip(all_content_latents, content_path):
        for style_latents, style_file in zip(all_style_latents, style_path):
            print(f"Transferring style from {style_file.name} to {content_file.name}")
            
            generated_images_list = pnp.run_pnp(content_latents, style_latents, style_file, content_fn=content_file, style_fn=style_file)
            generated_image_pil = generated_images_list[0]
            torch.cuda.empty_cache()

            content_fn_base = os.path.splitext(os.path.basename(content_file))[0]
            style_fn_base = os.path.splitext(os.path.basename(style_file))[0]
            
            
            if is_tileable:
                print("Blending Image Seams...")
                #integrate seam blending
                #convert PIL image to cv2 format
                im_np = np.array(generated_image_pil.convert('RGB'))
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
                #Save the blended image
                out_fn = f'{opt.prefix_name}{content_fn_base}_s{style_fn_base}_tiled.png'
                save_path = os.path.join(opt.output_dir, out_fn)
                #covnert rgb numpy back to bgr for opencv saving
                cv2.imwrite(save_path, cv2.cvtColor(final_im_blended, cv2.COLOR_RGB2BGR))
                print(f"Saved final blended image to {save_path}")

            else:
                out_fn = f'{opt.prefix_name}{content_fn_base}_s{style_fn_base}_raw.png'
                save_path = os.path.join(opt.output_dir, out_fn)
                generated_image_pil.save(save_path) # Use PIL's save method for the raw image
                print(f"Saved raw generated image to {save_path}")
            

    if gen_normal:
        print("Cleaning up Style Transfer model to free VRAM for Marigold...")
        del blip_diffusion_pipe
        del pnp
        torch.cuda.empty_cache()
        print("Enabling Marigold Pipe for Normal Generation...")
        from diffusers import MarigoldNormalsPipeline

        marigold_pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v1-1",
            variant="fp16",
            torch_dtype=torch.float16
        ).to("cuda")
    
        output_images = list(Path(opt.output_dir).glob("*_raw.png")) + \
                        list(Path(opt.output_dir).glob("*_tiled.png"))
        #normal map generation
        for img_path in output_images:
            input_img = Image.open(img_path).convert("RGB")
            final_normal = generate_normal(input_img, marigold_pipe)

            #Save the normal map
            out_fn = f'{opt.prefix_name}{content_fn_base}_s{style_fn_base}_normal.png'
            save_path = os.path.join(opt.output_dir, out_fn)
            final_normal[0].save(save_path)
            print(f"Saved final blended image to {save_path}")
            
            


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

def apply_seam_blending(image,gap_px,blur,min_ratio,im_origin_size=None,maintain_size=True):
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

def generate_normal(image, pipe):

    #load image
    normals = pipe(image)
    final_im = pipe.image_processor.visualize_normals(normals.prediction)
    return final_im

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

    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    parser.add_argument('--prefix_name', type=str, default='')
    
    #additional arguments for blending seams
    parser.add_argument('--gap',type=float, default=0.12, help="Size of border that will be used for blending in pixel size or in percentage if under 1")
    parser.add_argument('--min_ratio',type=float, default=0.2, help='Used to ensure balanced blending')
    parser.add_argument('--blurring',type=int, default=3,choices=range(1,10,2), help="Size of Gaussian blur to make merging softer.Use odd numbers only.")
    parser.add_argument('--maintain_size',type=bool, default=True,help="maintain images default size")

    opt = parser.parse_args()

    run(opt)

