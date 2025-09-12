import torch
import torchvision
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from parse_config import cfg
from data_loader.loader import ContentData
import torch.nn.functional as F


class SimpleTrainer:
    """Simple trainer for single-device training (CPU or single GPU)"""
    
    def __init__(self, diffusion, model, vae, criterion, optimizer, data_loader, logs, valid_data_loader, device):
        self.diffusion = diffusion
        self.model = model
        self.vae = vae
        self.nce_criterion = criterion['nce']
        self.recon_criterion = criterion['recon']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        
        # Setup logging directories from logs dict
        self.log_dir = logs['tboard'] if isinstance(logs, dict) else logs
        self.checkpoint_dir = logs['model'] if isinstance(logs, dict) else os.path.join(logs, "checkpoints")
        self.image_dir = logs['sample'] if isinstance(logs, dict) else os.path.join(logs, "images")
        
        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.tb_summary = SummaryWriter(log_dir=self.log_dir)
        
        print(f"📁 Logging to: {self.log_dir}")
        print(f"💾 Checkpoints: {self.checkpoint_dir}")
        print(f"🖼️  Images: {self.image_dir}")

    def _train_iter(self, data, step, pbar):
        self.model.train()
        
        # Prepare input
        images = data['img'].to(self.device)
        style_ref = data['style'].to(self.device)
        laplace_ref = data['laplace'].to(self.device)
        content_ref = data['content'].to(self.device)
        wid = data['wid'].to(self.device)
        
        # VAE encode
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215

        # Forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        
        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
        
        # Calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
            
        self.optimizer.step()

        # Logging
        loss_dict = {
            "reconstruct_loss": recon_loss.item(), 
            "high_nce_loss": high_nce_loss.item(),
            "low_nce_loss": low_nce_loss.item(),
            "total_loss": loss.item()
        }
        self.tb_summary.add_scalars("loss", loss_dict, step)
        
        # Update progress bar
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'NCE_H': f'{high_nce_loss.item():.4f}',
                'NCE_L': f'{low_nce_loss.item():.4f}'
            })

        return loss.item()

    @torch.no_grad()
    def _validate(self, epoch):
        """Simple validation with image generation"""
        print(f"🔍 Running validation for epoch {epoch}...")
        self.model.eval()
        
        try:
            # Get a batch from validation data
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
            
            # Prepare input
            style_ref = test_data['style'].to(self.device)
            laplace_ref = test_data['laplace'].to(self.device)
            
            # Load content for text generation
            load_content = ContentData()
            texts = ['hello', 'world', 'test']
            
            generated_images = []
            
            for i, text in enumerate(texts[:min(3, style_ref.shape[0])]):
                text_ref = load_content.get_content(text)
                text_ref = text_ref.to(self.device).unsqueeze(0)
                
                # Generate random noise
                h, w = style_ref.shape[2], text_ref.shape[1] * 32
                x = torch.randn((1, 4, h//8, w//8)).to(self.device)
                
                # Generate image using DDIM
                generated_latent = self.diffusion.ddim_sample(
                    self.model, x, 
                    style_ref[i:i+1], laplace_ref[i:i+1], text_ref,
                    sampling_timesteps=20
                )
                
                # Decode with VAE
                generated_image = self.vae.decode(generated_latent / 0.18215).sample
                generated_images.append(generated_image)
            
            if generated_images:
                # Save generated images
                all_images = torch.cat(generated_images, dim=0)
                grid = torchvision.utils.make_grid(all_images, normalize=True, scale_each=True)
                
                # Save to file
                image_path = os.path.join(self.image_dir, f"epoch_{epoch:04d}.png")
                torchvision.utils.save_image(grid, image_path)
                
                # Log to tensorboard
                self.tb_summary.add_image("generated_samples", grid, epoch)
                print(f"💾 Saved validation images to {image_path}")
                
        except Exception as e:
            print(f"⚠️  Validation failed: {e}")
            print("Continuing training...")

    def _save_checkpoint(self, epoch, step):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print("🚀 Starting training loop...")
        
        total_steps = 0
        
        for epoch in range(cfg.SOLVER.EPOCHS):
            print(f"\n📅 Epoch {epoch}/{cfg.SOLVER.EPOCHS}")
            
            # Training
            self.model.train()
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for step, data in enumerate(pbar):
                loss = self._train_iter(data, total_steps, pbar)
                epoch_loss += loss
                num_batches += 1
                total_steps += 1
                
                # Validation
                if (total_steps >= cfg.TRAIN.VALIDATE_BEGIN and 
                    total_steps % cfg.TRAIN.VALIDATE_ITERS == 0):
                    self._validate(epoch)
                
                # Save checkpoint
                if (total_steps >= cfg.TRAIN.SNAPSHOT_BEGIN and 
                    total_steps % cfg.TRAIN.SNAPSHOT_ITERS == 0):
                    self._save_checkpoint(epoch, total_steps)
            
            # Log epoch statistics
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"📊 Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            self.tb_summary.add_scalar("epoch_loss", avg_loss, epoch)
            
            # Save checkpoint at end of epoch
            if epoch % 5 == 0 or epoch == cfg.SOLVER.EPOCHS - 1:
                self._save_checkpoint(epoch, total_steps)
        
        print("✅ Training completed!")
        self.tb_summary.close()
