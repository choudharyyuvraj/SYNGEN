# ...existing code...
import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import ContentData
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed
import cv2
import random
import numpy as np
from torch.utils.data import Dataset


class CustomStyleDataset(Dataset):
    """
    Custom dataset for handwriting style images
    
    Expected folder structure:
    style_folder/
        writer1/
            sample1.png
            sample2.png
            ...
        writer2/
            sample1.png
            sample2.png
            ...
    """
    def __init__(self, style_folder, num_samples):
        self.style_folder = style_folder
        self.num_samples = num_samples
        
        # Get all writer folders
        self.writers = []
        if os.path.exists(style_folder):
            self.writers = [w for w in os.listdir(style_folder) 
                          if os.path.isdir(os.path.join(style_folder, w))]
        
        if not self.writers:
            raise ValueError(f"No writer folders found in {style_folder}")
            
        print(f"Found {len(self.writers)} writers: {self.writers}")
    
    def __len__(self):
        return self.num_samples
    
    def get_style_image(self, writer_id):
        """Get a random style image from a writer"""
        writer_path = os.path.join(self.style_folder, writer_id)
        images = [f for f in os.listdir(writer_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            raise ValueError(f"No images found for writer {writer_id}")
            
        # Select random image
        image_file = random.choice(images)
        image_path = os.path.join(writer_path, image_file)
        
        # Load and process image
        style_image = cv2.imread(image_path, flags=0)  # Load as grayscale
        
        if style_image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Ensure minimum width
        if style_image.shape[1] < 128:
            print(f"Warning: Image {image_file} is too narrow ({style_image.shape[1]}px), using anyway")
        
        # Normalize to [0, 1]
        style_image = style_image / 255.0
        
        # Create laplace edge detection
        laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplace_image = cv2.filter2D(style_image, -1, laplace_kernel)
        laplace_image = np.abs(laplace_image)
        
        return style_image, laplace_image
    
    def __getitem__(self, idx):
        batch = []
        
        for writer_id in self.writers:
            try:
                style_ref, laplace_ref = self.get_style_image(writer_id)
                
                # Convert to tensors
                style_ref = torch.from_numpy(style_ref).unsqueeze(0).float()
                laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0).float()
                
                batch.append({
                    'style': style_ref,
                    'laplace': laplace_ref,
                    'wid': writer_id
                })
            except Exception as e:
                print(f"Error processing writer {writer_id}: {e}")
                continue
        
        if not batch:
            raise ValueError("No valid style images found")
        
        # Pad to same width for batching
        max_width = max([item['style'].shape[2] for item in batch])
        
        for item in batch:
            current_width = item['style'].shape[2]
            if current_width < max_width:
                # Pad to max width
                pad_width = max_width - current_width
                item['style'] = torch.nn.functional.pad(item['style'], (0, pad_width), value=1.0)
                item['laplace'] = torch.nn.functional.pad(item['laplace'], (0, pad_width), value=0.0)
        
        # Stack into batch tensors
        style_batch = torch.stack([item['style'] for item in batch])
        laplace_batch = torch.stack([item['laplace'] for item in batch])
        wid_batch = [item['wid'] for item in batch]
        
        return {
            'style': style_batch,
            'laplace': laplace_batch,
            'wid': wid_batch
        }


def load_custom_texts(text_source):
    """
    Load text from various sources:
    - If it's a file path, read words from file
    - If it's a string with commas, split by commas
    - If it's a single string, use as-is
    """
    if os.path.isfile(text_source):
        # Load from file
        with open(text_source, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if ',' in content:
                texts = [t.strip() for t in content.split(',') if t.strip()]
            else:
                texts = content.split()
    elif ',' in text_source:
        # Split by commas
        texts = [t.strip() for t in text_source.split(',') if t.strip()]
    else:
        # Single word or sentence
        texts = [text_source.strip()]
    
    return texts


def main(opt):
    """Load config file into cfg"""
    if os.path.exists(opt.cfg_file):
        cfg_from_file(opt.cfg_file)
        assert_and_infer_cfg()
    else:
        print(f"Warning: Config file {opt.cfg_file} not found, using defaults")
    
    """Fix the random seed"""
    fix_seed(42)  # Default seed

    # Single device setup
    device = torch.device(opt.device)
    print(f"Using device: {device}")

    # Load content processor
    load_content = ContentData()

    # Load custom texts
    print(f"Loading texts from: {opt.text_source}")
    texts = load_custom_texts(opt.text_source)
    print(f"Found {len(texts)} texts to generate: {texts}")
    
    # Limit number of texts for testing
    if opt.max_texts > 0:
        texts = texts[:opt.max_texts]
        print(f"Limited to first {len(texts)} texts")
    
    # Setup custom style dataset
    print(f"Loading style images from: {opt.style_folder}")
    style_dataset = CustomStyleDataset(opt.style_folder, len(texts))
    
    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    # Create output directory
    os.makedirs(opt.save_dir, exist_ok=True)
    print(f"Output directory: {opt.save_dir}")

    # Initialize diffusion
    diffusion = Diffusion(device=device)

    # Build model architecture
    print("Building UNet model...")
    unet = UNetModel(
        in_channels=4,  # Default from config
        model_channels=128,  # Default from config
        out_channels=4,  # Default from config
        num_res_blocks=2,  # Default from config
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=4,  # Default from config
        context_dim=128  # Default from config
    ).to(device)
    
    # Load pretrained model
    if os.path.exists(opt.one_dm):
        print(f"Loading model from: {opt.one_dm}")
        checkpoint = torch.load(opt.one_dm, map_location=device)
        unet.load_state_dict(checkpoint)
        print('Successfully loaded pretrained One-DM model')
    else:
        raise IOError(f'Model checkpoint not found: {opt.one_dm}')
    unet.eval()

    # Load VAE
    try:
        print("Loading VAE from Stable Diffusion...")
        vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
        vae = vae.to(device)
        vae.requires_grad_(False)
        print("VAE loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load VAE from {opt.stable_dif_path}: {e}")
        print("Using dummy VAE for testing...")
        
        class DummyVAE:
            def decode(self, latents):
                latents = latents / 0.18215
                upsampled = torch.nn.functional.interpolate(latents, scale_factor=8, mode='bilinear')
                if upsampled.shape[1] == 4:
                    upsampled = upsampled.mean(dim=1, keepdim=True)
                return type('obj', (object,), {'sample': upsampled})()
        
        vae = DummyVAE()

    # Generate handwriting
    loader_iter = iter(style_loader)
    
    print(f"\nStarting generation for {len(texts)} texts...")
    print(f"Sample method: {opt.sample_method}")
    print(f"Sampling timesteps: {opt.sampling_timesteps}")
    
    for i, text in enumerate(tqdm(texts, position=0, desc='Generating handwriting')):
        try:
            # Get style data
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(style_loader)
                data = next(loader_iter)
                
            style_images = data['style'][0]  # Remove batch dimension
            laplace_images = data['laplace'][0]
            writer_ids = data['wid'][0]
            
            # Process each writer style
            for j in range(len(writer_ids)):
                try:
                    style_input = style_images[j:j+1].to(device)  # Keep batch dimension
                    laplace = laplace_images[j:j+1].to(device)
                    writer_id = writer_ids[j]
                    
                    # Get text content encoding
                    text_ref = load_content.get_content(text)
                    text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
                    
                    # Create random noise
                    noise_height = style_input.shape[2] // 8
                    noise_width = (text_ref.shape[1] * 32) // 8
                    x = torch.randn((text_ref.shape[0], 4, noise_height, noise_width)).to(device)
                    
                    # Sample using diffusion
                    if opt.sample_method == 'ddim':
                        generated_images = diffusion.ddim_sample(
                            unet, vae, style_input.shape[0], 
                            x, style_input, laplace, text_ref,
                            opt.sampling_timesteps, opt.eta
                        )
                    elif opt.sample_method == 'ddpm':
                        generated_images = diffusion.ddpm_sample(
                            unet, vae, style_input.shape[0], 
                            x, style_input, laplace, text_ref
                        )
                    else:
                        raise ValueError(f'Unsupported sample method: {opt.sample_method}')
                    
                    # Save generated images
                    for k, generated_img in enumerate(generated_images):
                        try:
                            # Convert to PIL Image
                            pil_image = torchvision.transforms.ToPILImage()(generated_img)
                            grayscale_image = pil_image.convert("L")
                            
                            # Create output path
                            output_filename = f"{text}_{writer_id}_{k}.png"
                            output_path = os.path.join(opt.save_dir, output_filename)
                            
                            # Save image
                            grayscale_image.save(output_path)
                            
                            if (i + 1) % 5 == 0:
                                print(f"Generated {i+1}/{len(texts)}: '{text}' in style of {writer_id}")
                                
                        except Exception as save_error:
                            print(f"Error saving image for '{text}' (writer {writer_id}): {save_error}")
                            
                except Exception as gen_error:
                    print(f"Error generating '{text}' for writer {writer_ids[j] if j < len(writer_ids) else 'unknown'}: {gen_error}")
                    continue
                    
        except Exception as text_error:
            print(f"Error processing text '{text}': {text_error}")
            continue

    print(f"\nGeneration complete!")
    print(f"Results saved in: {opt.save_dir}")
    print(f"Generated files for {len(texts)} texts")


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Generate handwriting with custom dataset')
    
    # Model and config
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (optional)')
    parser.add_argument('--one_dm', dest='one_dm', default='model_zoo/One-DM-ckpt.pt', 
                        help='Path to pretrained One-DM model')
    
    # Custom dataset paths
    parser.add_argument('--style_folder', required=True, 
                        help='Folder containing style images organized by writer')
    parser.add_argument('--text_source', required=True,
                        help='Text source: file path, comma-separated words, or single text')
    
    # Output settings  
    parser.add_argument('--save_dir', default='Generated_Custom', 
                        help='Directory to save generated images')
    
    # Generation settings
    parser.add_argument('--device', type=str, default='cpu', help='Device for generation')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', 
                        help='Sampling method: ddim or ddpm')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--max_texts', type=int, default=0, 
                        help='Limit number of texts to generate (0 = no limit)')
    
    opt = parser.parse_args()
    
    print("=== One-DM Custom Dataset Generation ===")
    print(f"Style folder: {opt.style_folder}")
    print(f"Text source: {opt.text_source}")
    print(f"Device: {opt.device}")
    print(f"Model: {opt.one_dm}")
    print(f"Output: {opt.save_dir}")
    print(f"Max texts: {opt.max_texts if opt.max_texts > 0 else 'All'}")
    
    main(opt)
