import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from models.unet import UNetModel
from tqdm import tqdm
import torch
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed
from PIL import Image
import numpy as np

# Simple content data class for single image testing
class SimpleContentData:
    def __init__(self):
        pass
    
    def get_content(self, text):
        """Create content tensor from text"""
        # Create a simple text image
        img = Image.new('L', (256, 64), color=255)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Center text
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w, h = len(text) * 12, 16
        
        x = (256 - w) // 2
        y = (64 - h) // 2
        draw.text((x, y), text, fill=0, font=font)
        
        # Convert to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        
        content_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        return content_tensor

# Simple style dataset for single image
class SimpleStyleDataset:
    def __init__(self, style_image_path, num_samples=1):
        self.style_image_path = style_image_path
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load and process style image
        if os.path.exists(self.style_image_path):
            style_img = Image.open(self.style_image_path).convert('L')
        else:
            # Create dummy style if image doesn't exist
            style_img = Image.new('L', (352, 64), color=128)
        
        # Resize to expected style format
        style_img = style_img.resize((352, 64), Image.LANCZOS)
        
        # Convert to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        
        style_tensor = transform(style_img)
        
        # Create Laplace kernel for edge detection
        laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float32)
        laplace = laplace.view(1, 3, 3)
        
        # Create dummy writer ID
        wid = [f"writer_{idx}"]
        
        return {
            'style': [style_tensor.unsqueeze(0)],  # Add batch dimension
            'laplace': [laplace.unsqueeze(0)],     # Add batch dimension  
            'wid': wid
        }

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    # Remove distributed training setup for single image testing
    device = torch.device(opt.device)

    load_content = SimpleContentData()

    # Use provided text or default test words
    if hasattr(opt, 'test_text') and opt.test_text:
        temp_texts = opt.test_text.split(',')
    else:
        temp_texts = ["Hello", "World", "Test"]  # Default test words

    """setup data_loader instances"""
    style_dataset = SimpleStyleDataset(opt.style_image, len(temp_texts))
    
    print('Number of test words: ', len(temp_texts))
    print('Test words: ', temp_texts)
    
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                                batch_size=1,
                                                shuffle=False,  # Don't shuffle for consistent results
                                                drop_last=False,
                                                num_workers=0,  # Set to 0 for single image
                                                pin_memory=False
                                                )

    target_dir = os.path.join(opt.save_dir, "single_image_test")
    os.makedirs(target_dir, exist_ok=True)

    diffusion = Diffusion(device=device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
            checkpoint = torch.load(opt.one_dm, map_location=device)
            if "model_state_dict" in checkpoint:
                unet.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                unet.load_state_dict(checkpoint["state_dict"])
            else:
                unet.load_state_dict(checkpoint)
            print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
            raise IOError('input the correct checkpoint path')
    unet.eval()

    # Load VAE
    try:
        vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
        vae = vae.to(device)
        vae.requires_grad_(False)
        print("Loaded VAE from Stable Diffusion")
    except Exception as e:
        print(f"Could not load VAE: {e}")
        print("Using simple VAE decoder")
        # Simple VAE fallback
        class SimpleVAE:
            def __init__(self, device):
                self.device = device
            
            def decode(self, latents):
                # Simple decoder
                latents = latents / 0.18215
                upsampled = torch.nn.functional.interpolate(
                    latents, scale_factor=8, mode='bilinear', align_corners=False
                )
                if upsampled.shape[1] == 4:
                    upsampled = upsampled.mean(dim=1, keepdim=True)
                return type('obj', (object,), {'sample': upsampled})()
        
        vae = SimpleVAE(device)

    """generate the handwriting datasets"""
    loader_iter = iter(style_loader)
    
    for i, x_text in enumerate(tqdm(temp_texts, desc='Generating text')):
        try:
            data = next(loader_iter)
        except StopIteration:
            # If we run out of style data, restart the iterator
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
            
        for (data_val, laplace, wid) in data_loader:
            style_input = data_val.to(device)
            laplace = laplace.to(device)
            
            # Get content for current text
            text_ref = load_content.get_content(x_text)
            text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
            
            # Create random latent
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(device)
            
            print(f"Generating '{x_text}' with shapes:")
            print(f"  Style: {style_input.shape}")
            print(f"  Content: {text_ref.shape}")
            print(f"  Latent: {x.shape}")
            
            try:
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
                if isinstance(ema_sampled_images, torch.Tensor):
                    # Handle tensor output
                    for index in range(ema_sampled_images.shape[0]):
                        img_tensor = ema_sampled_images[index]
                        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                            img_tensor = img_tensor.squeeze(0)
                        
                        # Normalize to [0, 1]
                        img_tensor = (img_tensor + 1) / 2
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Convert to PIL
                        im = torchvision.transforms.ToPILImage()(img_tensor)
                        image = im.convert("L")
                        
                        out_path = os.path.join(target_dir, f"writer_{index}")
                        os.makedirs(out_path, exist_ok=True)
                        output_file = os.path.join(out_path, x_text + ".png")
                        image.save(output_file)
                        print(f"Saved: {output_file}")
                
                elif isinstance(ema_sampled_images, np.ndarray):
                    # Handle numpy array output
                    for index in range(len(ema_sampled_images)):
                        img_array = ema_sampled_images[index]
                        if img_array.ndim == 3:
                            img_array = img_array[0]  # Take first channel
                        
                        # Normalize to [0, 255]
                        img_array = (img_array * 255).astype(np.uint8)
                        image = Image.fromarray(img_array, mode='L')
                        
                        out_path = os.path.join(target_dir, f"writer_{index}")
                        os.makedirs(out_path, exist_ok=True)
                        output_file = os.path.join(out_path, x_text + ".png")
                        image.save(output_file)
                        print(f"Saved: {output_file}")
                
                else:
                    # Handle list output (original format)
                    for index in range(len(ema_sampled_images)):
                        im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                        image = im.convert("L")
                        out_path = os.path.join(target_dir, wid[index % len(wid)])
                        os.makedirs(out_path, exist_ok=True)
                        output_file = os.path.join(out_path, x_text + ".png")
                        image.save(output_file)
                        print(f"Saved: {output_file}")
                        
            except Exception as e:
                print(f"Error generating '{x_text}': {e}")
                # Save content as fallback
                content_normalized = (text_ref[0] + 1) / 2
                content_normalized = torch.clamp(content_normalized, 0, 1)
                im = torchvision.transforms.ToPILImage()(content_normalized)
                image = im.convert("L")
                
                out_path = os.path.join(target_dir, "fallback")
                os.makedirs(out_path, exist_ok=True)
                output_file = os.path.join(out_path, x_text + "_fallback.png")
                image.save(output_file)
                print(f"Saved fallback: {output_file}")

    print(f"\nGeneration complete! Check {target_dir}/ for results.")

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', 
                        help='target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='model_zoo/One-DM-ckpt.pt', 
                        help='pre-train model for generating')
    parser.add_argument('--style_image', default='test.jpg', 
                        help='Style reference image path')
    parser.add_argument('--test_text', default='Hello,World,Test', 
                        help='Comma-separated text to generate (e.g., "Hello,World,Test")')
    parser.add_argument('--device', type=str, default='cpu', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=20)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    
    opt = parser.parse_args()
    main(opt)
