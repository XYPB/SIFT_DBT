import os
import time
import builtins
import json
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.backends import cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
from lightly.loss import NTXentLoss
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

from opt import get_opt
from dataset.BCS_DBT_cls_dataset import get_dataset
from dataset.sampler import get_sampler
from models.simple_cnn import SimpleCNN
from utils import manual_seed, get_log_dir, log_images_to_writer, calc_model_grad_norm

from models.moco import MoCo
from models.lars import LARS


def train(args, model, criterion, scaler, momentum_val, device, train_loader, optimizer, writer, epoch):
    model.train()
    st = time.time()
    step_losses = []
    min_scale = 128
    for batch_idx, (data, label) in enumerate(train_loader):
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        if isinstance(data, list):
            data = [d.to(device) for d in data]
        else:
            data = data.to(device)

        dtype = torch.bfloat16 if args.bf16 else None
        with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
            q, k = model(data, momentum_val=momentum_val)
            loss = criterion(q, k)
            loss /= args.accum_steps
        # autocast should only wrap the forward pass
        scaler.scale(loss).backward()
        if writer:
            if args.log_grad_norm:
                writer.add_scalar('Grad Norm/train', calc_model_grad_norm(model), global_step)

        if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler.get_scale())
            optimizer.zero_grad()

        #### Log results
        step_losses.append(loss.detach().cpu().numpy())
        if writer:
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Avg. Loss/train', np.mean(step_losses), global_step)
            if scaler.is_enabled():
                writer.add_scalar('Scaler/_scale', scaler._scale.item(), global_step)
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAvg. Loss: {np.mean(step_losses):.6f}')
            if writer and (batch_idx == 0 or args.dev):
                key_image = data[0].detach().cpu().numpy()
                log_images_to_writer(key_image, writer, global_step, args, 'key')
                query_image = data[1].detach().cpu().numpy()
                log_images_to_writer(query_image, writer, global_step, args, 'query')
            if args.dry_run:
                break
    et = time.time()
    print(f"Train Avg. Loss: {np.mean(step_losses):.4f},\tTime: {et-st:.2f}\n")
    return


def main_worker(rank, world_size, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.ddp:
        torch.cuda.set_device(rank)
        print(f"### Use GPU:{rank} for training")
        setup(rank, world_size, args)

        # suppress printing if not master
        if rank != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass
    
    manual_seed(args.seed)
    if args.save_model or args.log:
        log_dir, config_dir = get_log_dir(args)
    else:
        log_dir = None

    #### GPU args
    if use_cuda:
        if args.ddp:
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        if args.num_workers == -1:
            args.num_workers = len(os.sched_getaffinity(0))
        print(f'### Use {args.num_workers} cores for training...')
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)

    if args.cudnn:
        cudnn.benchmark = True

    #### Data
    norm = {"mean": [0.1550], "std": [0.1521]}
    transform = MoCoV2Transform(input_size=args.target_H, min_scale=0.4,
                                cj_strength=args.cj_strength, normalize=norm,
                                vf_prob=0., rr_prob=0.2,
                                rr_degrees=30)
    args.contrastive = True
    train_dataset = get_dataset('data/csv/BCS-DBT_train_label_v2.csv', transform, args)

    #### Weighted sampling
    if args.balance_data or args.binary_balance or args.ddp:
        train_sampler = get_sampler(train_dataset, world_size, rank, args)
        train_kwargs.pop('shuffle')
        train_kwargs['sampler'] = train_sampler

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)

    ### In-channels
    in_channels = args.num_slice
    if args.patchify:
        in_channels *= args.patch_cnt

    #### Model
    model_args = {}
    if 'simple' in args.model_type:
        model = SimpleCNN(in_channels, 10, base_channels=32, dropout_p=args.dropout_p)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    else:
        if 'resnet' in args.model_type:
            model = getattr(torchvision.models, args.model_type)(**model_args)
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
    model = MoCo(model, feat_chs=in_features)
    model = model.to(device)
    if args.ddp:
        model = DDP(model, device_ids=[rank])

    #### Scaler
    scaler = GradScaler(enabled=args.scaler, growth_interval=200)

    #### loss
    criterion = NTXentLoss(memory_bank_size=4096)

    #### Resume
    if args.resume is not None:
        print(f'### resume experiment logged under {args.resume}...')
        log_dir = args.resume
        config_dir = os.path.join(log_dir, 'config')
        ckpt_dir = os.path.join(log_dir, 'ckpt')
        assert os.path.exists(log_dir)
        assert os.path.exists(config_dir)
        assert os.path.exists(ckpt_dir)
        ckpts = sorted(glob(os.path.join(ckpt_dir, '*.ckpt')))
        ckpt_path = ckpts[-1]
        print(f'### Load model from {ckpt_path}...')
        state_dict = torch.load(ckpt_path)
        # Get current epoch
        if args.cur_ep is not None:
            cur_ep = args.cur_ep
        elif 'epoch' in state_dict.keys():
            cur_ep = state_dict['epoch']
        else:
            cur_ep = int(ckpt_path.split('_ep')[-1].replace('.ckpt', ''))
        # load scaler
        if 'scaler' in state_dict.keys():
            scaler.load_state_dict(state_dict['scaler'])
        # load model
        if 'weight' in state_dict.keys() and len(state_dict) == 3:
            state_dict = state_dict['weight']
        model.load_state_dict(state_dict)
        print(f'### resume scheduler from {cur_ep}, target at {args.epochs}')
    elif args.log or args.save_model:
        try:
            if args.ddp:
                if rank == 0:
                    os.makedirs(log_dir, exist_ok=False)
            else:
                os.makedirs(log_dir, exist_ok=False)
        except FileExistsError as e:
            time.sleep(5)
            log_dir, config_dir = get_log_dir(args)
            os.makedirs(log_dir, exist_ok=False)
        print(f'### experiment logged under {log_dir}...')
        os.makedirs(config_dir, exist_ok=True)
        arg_dict = vars(args)
        json.dump(arg_dict, open(os.path.join(config_dir, 'train_config.json'), 'w'))
        cur_ep = 1
    else:
        cur_ep = 1

    #### Optimizer
    eff_bsz = args.batch_size * world_size
    eff_lr = args.lr * eff_bsz / 256
    print(f'### Effective learning rate: {eff_lr:.5f} with base lr: {args.lr:.5f}')
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    elif args.lars:
        optimizer = LARS(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay,
                         momentum=args.momentum)
    elif args.sgd:
        optimizer = optim.SGD(model.parameters(), lr=eff_lr, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    elif args.adamW:
        optimizer = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=eff_lr)

    torch.autograd.set_detect_anomaly(True)

    #### Scheduler
    if cur_ep-2 != -1:
        # init initial_lr
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', eff_lr)
    if args.cos:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=cur_ep-2)
    elif args.step:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, last_epoch=cur_ep-2)
    elif args.multi_step:
        scheduler = MultiStepLR(optimizer, [50, 75], gamma=0.1, last_epoch=cur_ep-2)
    else:
        scheduler = LambdaLR(optimizer, lambda x: x, last_epoch=cur_ep-2)

    if rank == 0 and args.log:
        # Create a SummaryWriter object for TensorBoard
        if args.log:
            runs_dir = os.path.join(log_dir, f'runs_ep{cur_ep:0>5d}')
            os.makedirs(runs_dir, exist_ok=True)
            writer = SummaryWriter(runs_dir)
    else:
        writer = None

    #### Train loop
    for epoch in range(cur_ep, args.epochs + 1):
        momentum_val = cosine_schedule(epoch-1, args.epochs, 0.996, 1)
        train(
            args, model, criterion, scaler, momentum_val, device, 
            train_loader, optimizer, writer, epoch,
        )
        if epoch == cur_ep:
            print(torch.cuda.memory_summary(device=device, abbreviated=False))
        if writer:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch - 1)
            writer.add_scalar('moco momentum', momentum_val, epoch - 1)
        scheduler.step()

        if args.ddp and rank != 0:
            time.sleep(1)
            continue

        if args.save_model:
            ckpt_dir = os.path.join(log_dir, 'ckpt')
            os.makedirs(ckpt_dir, exist_ok=True)
            state_dict = {
                'weight': model.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
            }
            if args.save_rec and ((epoch + 1) % args.save_interval) == 0:
                torch.save(state_dict, os.path.join(ckpt_dir, f"{args.model_type}_ep{epoch:0>4d}.ckpt"))
            else:
                ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                torch.save(state_dict, ckpt_dest)

    if args.ddp:
        cleanup()

    if args.log and writer:
        writer.flush()
        writer.close()


def setup(rank, world_size, args):
    # initialize the process group
    dist.init_process_group("nccl", init_method=args.dist_url, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_opt()
    if args.ddp:
        world_size = args.world_size
        torch.multiprocessing.spawn(main_worker, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)