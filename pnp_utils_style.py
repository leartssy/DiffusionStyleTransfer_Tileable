import torch
import os
import random
import numpy as np
#added
from pathlib import Path
from tqdm import tqdm
##

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    # conv_module = model.unet.up_blocks[1].resnets[1]
    # setattr(conv_module, 't', t)
    res_dict = {0: [0,1,2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

#added code for faster loading:
def load_or_aggregate_latents(save_path: Path, ddpm_steps: int, num_style_steps: int = None):
    """
    Attempts to load the single aggregated latent file first.
    If not found, falls back to loading and aggregating the individual step files.
    
    Args:
        save_path (Path): Path to the folder containing the noisy_latents_*.pt files.
        ddpm_steps (int): Total number of diffusion steps (e.g., 999).
        num_style_steps (int): If provided, limits the number of steps loaded (for style latents).
    
    Returns:
        torch.Tensor or None: The aggregated latents tensor, or None if unsuccessful.
    """
    aggregated_path = save_path / 'aggregated_latents.pt'
    
    # --- 1. Attempt Fast Load ---
    if aggregated_path.exists():
        print(f"Loading aggregated latents from {save_path.name} (FAST)")
        return torch.load(aggregated_path, map_location='cuda')

    # --- 2. Fallback to Slow Aggregation ---
    
    # Determine the actual number of steps to load
    max_steps = num_style_steps if num_style_steps is not None else ddpm_steps
    
    print(f"Loading individual latents for {save_path.name} (Slow I/O fallback)...")
    
    latent_list = []
    # Use trange for a progress bar during the slow load
    for t in tqdm(range(max_steps), desc=f"Aggregating {save_path.name} Steps"):
        latent_path = save_path / f'noisy_latents_{t}.pt'
        
        if latent_path.exists():
            # Load to CPU, then move to CUDA after concatenation
            latent_list.append(torch.load(latent_path, map_location='cpu')) 
        else:
            # We assume a contiguous sequence, so break if a file is missing
            break 
            
    if not latent_list:
        print(f"Warning: Could not load any latents in {save_path.name}.")
        return None

    # Concatenate, then move the whole thing to the GPU (CUDA) once
    combined_latents = torch.cat(latent_list, dim=0).to('cuda')
    print(f"Loaded {len(latent_list)} steps. Saving aggregate file now...")

    # --- 3. Save Aggregated File for Future Fast Loads ---
    # Save the consolidated file back to disk for next time!
    torch.save(combined_latents.cpu(), aggregated_path)
    
    return combined_latents
    ##


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_attention_control_efficient(model, injection_schedule, attention_weight=1.0):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            source_batch_size = x.shape[0] // 3
            if not is_cross and self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):

                # FAST: Project Q and K for source only and repeat (Locks Structure)
                q = self.to_q(x[:source_batch_size]).repeat(3, 1, 1)
                k = self.to_k(x[:source_batch_size]).repeat(3, 1, 1)
                
                # SLOW: Project V for the WHOLE batch (Allows Color Freedom)
                v = self.to_v(x) 
            else:
                # Standard non-injection path
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)

            q, k, v = map(self.head_to_batch_dim, (q, k, v))

                #old logic:
                #if not is_cross and self.injection_schedule is not None and (
                #        self.t in self.injection_schedule or self.t == 1000):
                #    q = self.to_q(x)
                #    k = self.to_k(encoder_hidden_states)

                #    source_batch_size = int(q.shape[0] // 2)
                    # inject unconditional
                #    q[source_batch_size:2 * source_batch_size] = q[:source_batch_size] 
                #    k[source_batch_size:2 * source_batch_size] = k[:source_batch_size] 
                    # inject conditional
                #    q[2 * source_batch_size:] = q[:source_batch_size] 
                #    k[2 * source_batch_size:] = k[:source_batch_size]

                #    q = self.head_to_batch_dim(q)
                #    k = self.head_to_batch_dim(k)
                #   v = self.to_v(encoder_hidden_states)
                #   v = self.head_to_batch_dim(v)


                #else :
                #    q = self.to_q(x)
                #   k = self.to_k(x)
                #   v = self.to_v(x)
                #   w = 0.8
                #    source_batch_size = int(q.shape[0] // 3)
                    #第一部分content第二部分无条件的第三部分有条件的
                    # inject unconditional
                #   k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                #   v[source_batch_size:2 * source_batch_size] = v[:source_batch_size]
                    
                    
                    # inject conditional
                #  k[2 * source_batch_size:] = k[:source_batch_size]
                #   v[2 * source_batch_size:] = v[:source_batch_size]

                #    q = self.head_to_batch_dim(q)
                #    k = self.head_to_batch_dim(k)
                #   v = self.head_to_batch_dim(v)


            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_conv_control_efficient(model, injection_schedule, conv_weight=0.8):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                # Fixed temb repeat for dimensions
                source_bs = input_tensor.shape[0] // 3
                repeat_dims = [1] * temb.dim()
                repeat_dims[0] = 3
                temb_p = self.time_emb_proj(self.nonlinearity(temb))[:source_bs].repeat(*repeat_dims)[:, :, None, None]
                
                if self.time_embedding_norm == "default":
                    hidden_states = hidden_states + temb_p
                elif self.time_embedding_norm == "scale_shift":
                    scale, shift = torch.chunk(temb_p, 2, dim=1)
                    hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.norm2(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            # --- SELECTIVE INJECTION (The Style Fix) ---
            if self.injection_schedule is not None and (self.t in self.injection_schedule):
                source_batch_size = input_tensor.shape[0] // 3
                
                # Slot 1: Unconditional (Full layout injection)
                hidden_states[source_batch_size:2*source_batch_size] = hidden_states[:source_batch_size]
                
                # Slot 2: Conditional (Blend to keep colors)
                w = conv_weight * 0.75 # Lower weight to preserve style
                hidden_states[2*source_batch_size:] = (1 - w) * hidden_states[2*source_batch_size:] + w * hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            return (input_tensor + hidden_states) / self.output_scale_factor

        return forward

    # conv_module = model.unet.up_blocks[1].resnets[1]
    # conv_module.forward = conv_forward(conv_module)
    # setattr(conv_module, 'injection_schedule', injection_schedule)
    
    res_dict = {1: [0, 1, 2], 2: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            conv_module = model.unet.up_blocks[res].resnets[block]
    
            conv_module.forward = conv_forward(conv_module)
  
            setattr(conv_module, 'injection_schedule', injection_schedule)

        