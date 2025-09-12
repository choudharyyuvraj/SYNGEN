# ...existing code...
import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import IAMDataset
import torch
from trainer.trainer import Trainer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.loss import SupConLoss
from torch.utils.data import Subset
import numpy as np
import random


class IAMDataset5Percent(IAMDataset):
    """Modified IAM Dataset that uses only 5% of the data for faster training"""
    
    def __init__(self, image_path, style_path, laplace_path, type, content_type='unifont', max_len=9, subset_ratio=0.05):
        super().__init__(image_path, style_path, laplace_path, type, content_type, max_len)
        
        # Calculate 5% subset size
        original_size = len(self.indices)
        subset_size = max(1, int(original_size * subset_ratio))
        
        # Randomly sample 5% of the data
        random.seed(42)  # For reproducible results
        subset_indices = random.sample(range(original_size), subset_size)
        
        # Create new data dict with only 5% of data
        self.original_data_dict = self.data_dict.copy()
        self.data_dict = {i: self.original_data_dict[self.indices[idx]] for i, idx in enumerate(subset_indices)}
        self.indices = list(range(len(subset_indices)))
        
        print(f"📊 Dataset reduced from {original_size} to {len(self.indices)} samples ({subset_ratio*100:.1f}%)")


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set device (support both single GPU and CPU) """
    if opt.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA for training")
        
        # Initialize distributed training if multiple GPUs
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
            local_rank = dist.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            use_ddp = True
        else:
            local_rank = 0
            use_ddp = False
    else:
        device = torch.device('cpu')
        local_rank = 0
        use_ddp = False
        print("💻 Using CPU for training")

    """ set dataset with 5% subset """
    print("📚 Loading training dataset (5% subset)...")
    train_dataset = IAMDataset5Percent(
        cfg.DATA_LOADER.IAMGE_PATH, 
        cfg.DATA_LOADER.STYLE_PATH, 
        cfg.DATA_LOADER.LAPLACE_PATH, 
        cfg.TRAIN.TYPE,
        subset_ratio=opt.subset_ratio
    )
    print(f'🔢 Number of training images: {len(train_dataset)}')
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        drop_last=False,
        collate_fn=train_dataset.collate_fn_,
        num_workers=min(cfg.DATA_LOADER.NUM_THREADS, 2),  # Reduce workers for small dataset
        pin_memory=True if device.type == 'cuda' else False,
        sampler=train_sampler,
        shuffle=train_sampler is None  # Shuffle if not using sampler
    )
    
    print("📚 Loading test dataset (5% subset)...")
    test_dataset = IAMDataset5Percent(
        cfg.DATA_LOADER.IAMGE_PATH, 
        cfg.DATA_LOADER.STYLE_PATH, 
        cfg.DATA_LOADER.LAPLACE_PATH, 
        cfg.TEST.TYPE,
        subset_ratio=opt.subset_ratio
    )
    
    if use_ddp:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        drop_last=False,
        collate_fn=test_dataset.collate_fn_,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=min(cfg.DATA_LOADER.NUM_THREADS, 2),
        sampler=test_sampler,
        shuffle=False
    )
    
    """build model architecture"""
    print("🏗️ Building model architecture...")
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS, 
        model_channels=cfg.MODEL.EMB_DIM, 
        out_channels=cfg.MODEL.OUT_CHANNELS, 
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
        attention_resolutions=(1,1), 
        channel_mult=(1, 1), 
        num_heads=cfg.MODEL.NUM_HEADS, 
        context_dim=cfg.MODEL.EMB_DIM
    ).to(device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0:
        print(f"📥 Loading pretrained One-DM model from {opt.one_dm}")
        unet.load_state_dict(torch.load(opt.one_dm, map_location='cpu'))

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        print(f"📥 Loading pretrained ResNet18 model from {opt.feat_model}")
        checkpoint = torch.load(opt.feat_model, map_location='cpu')
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = unet.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "Failed to load the pretrained model"
    
    """Initialize the U-Net model for parallel training if using multiple GPUs"""
    if use_ddp:
        unet = DDP(unet, device_ids=[local_rank])
        
    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    
    # Adjust learning rate for smaller dataset
    adjusted_lr = cfg.SOLVER.BASE_LR * opt.lr_scale
    optimizer = optim.AdamW(unet.parameters(), lr=adjusted_lr)
    print(f"📈 Using learning rate: {adjusted_lr}")

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    print("📥 Loading VAE from Stable Diffusion...")
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae"""
    vae.requires_grad_(False)
    vae = vae.to(device)

    """build trainer"""
    print("🎯 Starting training...")
    
    # Create a simple trainer for single-device training
    from trainer.simple_trainer import SimpleTrainer
    trainer = SimpleTrainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    trainer.train()


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train One-DM model on 5% of data for faster experimentation')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', 
                        help='Path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64_scratch.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--feat_model', dest='feat_model', default='model_zoo/RN18_class_10400.pth', 
                        help='Pre-trained resnet18 model')
    parser.add_argument('--one_dm', dest='one_dm', default='', 
                        help='Pre-trained one_dm model (optional)')
    parser.add_argument('--log', default='5percent_training',
                        dest='log_name', required=False, help='The filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, 
                        help='Control the strength of noise')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device for training (cuda/cpu)')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='Local rank for distributed training')
    parser.add_argument('--subset_ratio', type=float, default=0.05, 
                        help='Ratio of data to use (default: 0.05 = 5%)')
    parser.add_argument('--lr_scale', type=float, default=1.0, 
                        help='Learning rate scaling factor for small dataset')
    
    opt = parser.parse_args()
    
    print("🚀 One-DM Training on 5% Data")
    print("=" * 50)
    print(f"📊 Dataset ratio: {opt.subset_ratio*100:.1f}%")
    print(f"🖥️  Device: {opt.device}")
    print(f"📝 Log name: {opt.log_name}")
    print(f"⚙️  Config: {opt.cfg_file}")
    print("=" * 50)
    
    main(opt)
