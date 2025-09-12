# ...existing code...
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from models.unet import UNetModel
from models.diffusion import Diffusion
from diffusers import AutoencoderKL
import os

# --- Helper: Load and preprocess image ---
def load_image(image_path, img_size=(64, 64)):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# --- Main inference ---
def main(opt):
    device = torch.device(opt.device)
    print(f"Loading model from {opt.model_checkpoint}")
    checkpoint = torch.load(opt.model_checkpoint, map_location=device)

    # Build model
    unet = UNetModel(
        in_channels=4,
        model_channels=512,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=(1,1),
        channel_mult=(1, 1),
        num_heads=4,
        context_dim=512
    ).to(device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()

    # Load VAE
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae").to(device)
    vae.eval()

    # Load image
    img = load_image(opt.image_path, img_size=(64, 64)).to(device)
    print(f"Image loaded: {img.shape}")

    # Encode image with VAE
    with torch.no_grad():
        latent = vae.encode(img).latent_dist.sample() * 0.18215

    # Prepare dummy style/content (for demo)
    style = torch.randn(1, 1, 64, 352).to(device)
    laplace = torch.randn(1, 1, 64, 352).to(device)
    content = torch.randn(1, 1, 64, 32).to(device)

    # Diffusion
    diffusion = Diffusion(device=device)
    t = diffusion.sample_timesteps(1).to(device)
    x_t, noise = diffusion.noise_images(latent, t)
    with torch.no_grad():
        pred_noise, _, _ = unet(x_t, t, style, laplace, content, tag='test')
        # DDIM sample (for demo)
        generated_latent = diffusion.ddim_sample(unet, x_t, style, laplace, content, t, sampling_timesteps=20)
        generated_img = vae.decode(generated_latent / 0.18215).sample

    # Save result
    out_path = os.path.splitext(opt.image_path)[0] + '_generated.png'
    T.ToPILImage()(generated_img.squeeze(0).cpu().clamp(-1, 1) * 0.5 + 0.5).save(out_path)
    print(f"Generated image saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to your custom image')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', help='Stable Diffusion VAE path')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    opt = parser.parse_args()
    main(opt)
