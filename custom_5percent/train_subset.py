import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed
from utils.logger import set_log
from data_loader.loader import IAMDataset
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion
from diffusers import AutoencoderKL
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from models.loss import SupConLoss
import os


def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    distributed = opt.distributed
    if distributed:
        # Choose backend based on platform
        backend = opt.backend
        if os.name == 'nt' and backend == 'nccl':
            backend = 'gloo'
        dist.init_process_group(backend=backend)
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(opt.device if torch.cuda.is_available() else 'cpu', local_rank if torch.cuda.is_available() else 0)
    else:
        local_rank = 0
        device = torch.device(opt.device if torch.cuda.is_available() and opt.device.startswith('cuda') else 'cpu')

    # Build full datasets
    full_train_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE
    )
    test_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TEST.TYPE
    )

    # Create a deterministic subset for training
    total = len(full_train_dataset)
    keep = max(1, int(total * opt.fraction))
    # Use a deterministic permutation based on cfg.TRAIN.SEED
    g = torch.Generator()
    g.manual_seed(cfg.TRAIN.SEED)
    perm = torch.randperm(total, generator=g).tolist()
    keep_indices = perm[:keep]
    train_dataset = Subset(full_train_dataset, keep_indices)

    # Samplers & Dataloaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.IMS_PER_BATCH,
            drop_last=False,
            collate_fn=full_train_dataset.collate_fn_,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            pin_memory=True,
            sampler=train_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            drop_last=False,
            collate_fn=test_dataset.collate_fn_,
            pin_memory=True,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            sampler=test_sampler,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.IMS_PER_BATCH,
            drop_last=False,
            collate_fn=full_train_dataset.collate_fn_,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            pin_memory=True,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            drop_last=False,
            collate_fn=test_dataset.collate_fn_,
            pin_memory=True,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            shuffle=False,
        )

    # Model
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM,
    ).to(device)

    # Optionally load checkpoints
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))

    # DDP (optional)
    if distributed and device.type == 'cuda':
        unet = nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank])

    # Criterion, Optimizer, Diffusion, VAE
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)

    # Trainer
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    print('Training on {:.2f}% of data: {}/{} samples'.format(opt.fraction * 100.0, keep, total))
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64_scratch.yml', help='Config file')
    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--log', default='debug', dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument('--distributed', action='store_true', help='enable distributed training')
    parser.add_argument('--backend', type=str, default='nccl', help='dist backend: nccl/gloo')
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction of training data to use (0,1]')
    opt = parser.parse_args()
    main(opt)


