import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    # Single device setup (no distributed training)
    device = torch.device(opt.device)

    load_content = ContentData()

    text_corpus = generate_type[opt.generate_type][1]
    print(f"Loading text corpus from: {text_corpus}")
    
    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()
    
    print(f"Found {len(texts)} words in corpus")
    
    # Limit number of words for testing (you can remove this limit)
    if hasattr(opt, 'max_words') and opt.max_words > 0:
        texts = texts[:opt.max_words]
        print(f"Limited to first {len(texts)} words for testing")
    
    temp_texts = texts
    
    """setup data_loader instances"""
    style_path = os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0])
    laplace_path = os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0])
    
    print(f"Style path: {style_path}")
    print(f"Laplace path: {laplace_path}")
    
    style_dataset = Random_StyleIAMDataset(style_path, laplace_path, len(temp_texts))
    
    print('Number of style samples: ', len(style_dataset))
    print('Number of words to generate: ', len(temp_texts))
    
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=0,  # Set to 0 for single process
                                                pin_memory=True
                                                )

    target_dir = os.path.join(opt.save_dir, opt.generate_type)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Output directory: {target_dir}")

    diffusion = Diffusion(device=device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        print(f"Loading model from: {opt.one_dm}")
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=device))
        print('Successfully loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('Please provide the correct checkpoint path with --one_dm')
    unet.eval()

    """Load VAE"""
    try:
        print("Loading VAE from Stable Diffusion...")
        vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
        vae = vae.to(device)
        vae.requires_grad_(False)
        print("VAE loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load VAE from {opt.stable_dif_path}: {e}")
        print("This might affect generation quality, but the script will continue...")
        
        # Create a dummy VAE for testing
        class DummyVAE:
            def decode(self, latents):
                latents = latents / 0.18215
                upsampled = torch.nn.functional.interpolate(latents, scale_factor=8, mode='bilinear')
                if upsampled.shape[1] == 4:
                    upsampled = upsampled.mean(dim=1, keepdim=True)
                return type('obj', (object,), {'sample': upsampled})()
        
        vae = DummyVAE()

    """generate the handwriting datasets"""
    loader_iter = iter(style_loader)
    
    print(f"\nStarting generation for {len(temp_texts)} words...")
    print(f"Sample method: {opt.sample_method}")
    print(f"Sampling timesteps: {opt.sampling_timesteps}")
    
    for i, x_text in enumerate(tqdm(temp_texts, position=0, desc='Generating words')):
        try:
            data = next(loader_iter)
        except StopIteration:
            # Reset iterator if we run out of style samples
            loader_iter = iter(style_loader)
            data = next(loader_iter)
            
        data_val, laplace, wid = data['style'][0], data['laplace'][0], data['wid']
        
        data_loader = []
        # split the data into two parts when the length of data is too large
        if len(data_val) > 224:
            data_loader.append((data_val[:224], laplace[:224], wid[:224]))
            data_loader.append((data_val[224:], laplace[224:], wid[224:]))
        else:
            data_loader.append((data_val, laplace, wid))
            
        for j, (data_val, laplace, wid) in enumerate(data_loader):
            try:
                style_input = data_val.to(device)
                laplace = laplace.to(device)
                text_ref = load_content.get_content(x_text)
                text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
                x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(device)
                
                if opt.sample_method == 'ddim':
                    ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                            x, style_input, laplace, text_ref,
                                                            opt.sampling_timesteps, opt.eta)
                elif opt.sample_method == 'ddpm':
                    ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
                                                            x, style_input, laplace, text_ref)
                else:
                    raise ValueError('sample method is not supported')
                
                # Save generated images
                for index in range(len(ema_sampled_images)):
                    try:
                        im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                        image = im.convert("L")
                        out_path = os.path.join(target_dir, wid[index][0])
                        os.makedirs(out_path, exist_ok=True)
                        output_file = os.path.join(out_path, x_text + ".png")
                        image.save(output_file)
                        
                        # Print progress every 10 words
                        if (i + 1) % 10 == 0:
                            print(f"Generated {i+1}/{len(temp_texts)}: {x_text} -> {output_file}")
                            
                    except Exception as save_error:
                        print(f"Error saving image for {x_text}: {save_error}")
                        
            except Exception as gen_error:
                print(f"Error generating {x_text}: {gen_error}")
                continue

    print(f"\nGeneration complete!")
    print(f"Results saved in: {target_dir}")
    print(f"Generated files for {len(temp_texts)} words")

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', 
                        help='target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='model_zoo/One-DM-ckpt.pt', 
                        help='pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True, 
                        help='four generation settings: iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cpu', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--max_words', type=int, default=0, 
                        help='Limit number of words to generate (0 = no limit)')
    
    opt = parser.parse_args()
    
    print("=== One-DM Test with IAM Dataset ===")
    print(f"Generate type: {opt.generate_type}")
    print(f"Device: {opt.device}")
    print(f"Model: {opt.one_dm}")
    print(f"Max words: {opt.max_words if opt.max_words > 0 else 'All'}")
    
    main(opt)
