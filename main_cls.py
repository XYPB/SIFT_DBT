# -*- coding: utf-8 -*-

import os
import time
import builtins
import json
import shutil
from glob import glob
from collections import Counter 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.backends import cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from opt import get_opt
from dataset.BCS_DBT_cls_dataset import get_dataset
from dataset.sampler import get_sampler
from dataset.transforms import get_transforms
from utils import (manual_seed, one_hot, get_log_dir, 
                   plot_auc_roc_img, log_images_to_writer, 
                   plot_auc_roc_img_multiclass,
                   disc_finetune_lr_dict, force_cudnn_initialization,
                   calc_model_grad_norm, get_specificity_with_sensitivity,
                   plot_confusion_matrix)
from models.simple_cnn import SimpleCNN



def train(args, model, device, train_loader, scaler, optimizer, writer, epoch, loss_weight, print_report=False):
    model.train()
    outputs = []
    targets = []
    step_losses = []
    min_scale = 128
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        if len(target.shape) > 1:
            target = target.squeeze(dim=1)
        data, target = data.to(device), target.to(device)
        dtype = torch.bfloat16 if args.bf16 else None
        with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
            output = model(data)
            if args.dev:
                print(output[0], target[0])
            elif args.loss_weight:
                loss = F.cross_entropy(output, target, weight=loss_weight)
            else:
                loss = F.cross_entropy(output, target)
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
        # cast torch bf16 to f32 for prediction output
        if args.bf16:
            output = output.detach().float()
        else:
            output = output.detach()

        #### Log results
        step_preds = np.argmax(output.cpu().numpy(), axis=1).squeeze()
        step_targets = target.detach().cpu().numpy()
        step_acc = 100 * accuracy_score(step_targets, step_preds)
        outputs.append(output.cpu().numpy())
        targets.append(step_targets)
        step_losses.append(loss.detach().cpu().numpy())
        if writer:
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Avg. Loss/train', np.mean(step_losses), global_step)
            writer.add_scalar('Batch Acc/train', step_acc, global_step)
            if scaler.is_enabled():
                writer.add_scalar('Scaler/_scale', scaler._scale.item(), global_step)
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAvg. Loss: {np.mean(step_losses):.6f}')
            if writer and batch_idx == 0:
                images = data.detach().cpu().numpy()
                log_images_to_writer(images, writer, global_step, args)
            if args.dry_run:
                break
    et = time.time()
    outputs = np.concatenate(outputs, axis=0)
    preds = np.argmax(outputs, axis=1).squeeze()
    targets = np.concatenate(targets, axis=0).squeeze()
    acc = 100 * accuracy_score(targets, preds)
    if args.binary:
        auc = 100 * roc_auc_score(targets, outputs[:, 1])
        specificity = 100 * get_specificity_with_sensitivity(targets, outputs[:, 1], 0.87)
    else:
        auc = 100 * roc_auc_score(one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class='ovr')
        specificity = 0.0
    if writer:
        if args.binary:
            auc_roc = plot_auc_roc_img(targets, outputs[:, 1], auc)
        else:
            auc_roc = plot_auc_roc_img_multiclass(targets, outputs, num_classes=outputs.shape[1])
        confusion_mat = plot_confusion_matrix(targets, preds, num_classes=outputs.shape[1],  normalize=False)
        writer.add_figure('Epoch AUC-ROC/train', auc_roc, global_step=epoch, close=True)
        writer.add_figure('Epoch Confusion Mat/train', confusion_mat, global_step=epoch, close=True)
    print(f"Train Acc: {acc:.2f}%,\tAUC: {auc:.2f}%,\tSpecificity: {specificity:.2f},\tTime: {et-st:.2f}")
    if print_report:
        print(classification_report(targets, preds, digits=4, zero_division=0.0))
    return acc, auc



def test(args, model, device, test_loader, writer, epoch, print_report=False, split='Test'):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            global_step = (epoch - 1) * len(test_loader) + batch_idx
            data, target = data.to(device), target.to(device)
            dtype = torch.bfloat16 if args.bf16 else None
            with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
                output = model(data)
                loss = F.cross_entropy(output, target.squeeze(), reduction='sum')
            if writer:
                writer.add_scalar('Loss/Test', loss.item(), global_step)
            test_loss += loss.item()  # sum up batch loss
            if args.bf16:
                output = output.detach().float()
            else:
                output = output.detach()
            outputs.append(output.cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    preds = np.argmax(outputs, axis=1).squeeze()
    targets = np.concatenate(targets, axis=0).squeeze()
    correct = np.sum(targets == preds)
    acc = 100 * accuracy_score(targets, preds)
    test_loss /= len(targets)
    if args.binary:
        auc = 100 * roc_auc_score(targets, outputs[:, 1])
        specificity = 100 * get_specificity_with_sensitivity(targets, outputs[:, 1], 0.87)
    else:
        auc = 100 * roc_auc_score(one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class='ovr')
        specificity = 0.0
    if writer:
        if args.binary:
            auc_roc = plot_auc_roc_img(targets, outputs[:, 1], auc)
        else:
            auc_roc = plot_auc_roc_img_multiclass(targets, outputs, num_classes=outputs.shape[1])
        confusion_mat = plot_confusion_matrix(targets, preds, num_classes=outputs.shape[1],  normalize=False)
        writer.add_figure(f'Epoch AUC-ROC/{split}', auc_roc, global_step=epoch, close=True)
        writer.add_figure(f'Epoch Confusion Mat/{split}', confusion_mat, global_step=epoch, close=True)

    print(f'\n{split} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(targets)} ({acc:.2f}%), AUC: {auc:.2f}%, Specificity: {specificity:.2f}\n')
    report_string = classification_report(targets, preds, digits=4, zero_division=0.0)
    if print_report:
        print(report_string)
    return acc, auc, report_string


def main_worker(rank, world_size, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.ddp:
        print(f"Use GPU:{rank} for training")
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
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        if args.num_workers == -1:
            args.num_workers = len(os.sched_getaffinity(0))
        print(f'### Use {args.num_workers} cores for training...')
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.cudnn:
        cudnn.benchmark = True
        force_cudnn_initialization()

    #### Data
    train_transform, test_transform = get_transforms(args)
    train_dataset = get_dataset('data/csv/BCS-DBT_train_label_v2.csv', train_transform, args)
    if args.test_patch:
        args.patch_lv = True
        args.patch_size = 448
        args.subset_ratio = 5.0
        args.subset = True
    val_dataset = get_dataset('data/csv/BCS-DBT_val_label_v2.csv', test_transform, args)
    test_dataset = get_dataset('data/csv/BCS-DBT_test_label_v2.csv', test_transform, args)

    #### Weighted sampling
    if args.balance_data or args.binary_balance or args.ddp:
        train_sampler = get_sampler(train_dataset, world_size, rank, args)
        train_kwargs.pop('shuffle')
        train_kwargs['sampler'] = train_sampler

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    #### Number of classes
    if args.binary:
        n_classes = 2
    elif args.vindr:
        n_classes = 5
    else:
        n_classes = 4
    if args.pretrain:
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights=None
    model_args = {
        'weights': weights,
        'num_classes': n_classes
    }

    #### loss weight
    if args.loss_weight:
        if args.patch_lv:
            y_train = [int(train_dataset.labels[i]) for i in range(len(train_dataset))]
        else:
            y_train = [int(train_dataset.labels[i]) for i in train_dataset.train_idx]
        if args.binary:
            if args.vindr:
                y_train = [0 if lb < 2 else 1 for lb in y_train]
            else:
                y_train = [0 if lb < 1 else 1 for lb in y_train]
        train_cnt = Counter(y_train)
        train_dist_list = [train_cnt[i] for i in range(len(train_cnt))]
        num_samples = sum(train_dist_list)
        train_weight = [num_samples / (n_classes * cnt) for cnt in train_dist_list]
        train_weight = torch.tensor(train_weight).to(device)
    else:
        train_weight = None

    ### In-channels
    in_channels = args.num_slice
    if args.patchify:
        in_channels *= args.patch_cnt

    #### Model
    if 'simple' in args.model_type:
        model = SimpleCNN(in_channels, model_args['num_classes'], base_channels=32, dropout_p=args.dropout_p)
    else:
        if 'resnet' in args.model_type:
            model = getattr(torchvision.models, args.model_type)(**model_args)
            in_features = model.fc.in_features
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise NotImplementedError

    model = model.to(device)

    #### Scaler
    scaler = GradScaler(enabled=args.scaler, growth_interval=200)

    ### Freeze backbone if liner probe
    if args.linear_probe:
        for name, param in model.named_parameters():
            if 'vit' in args.model_type:
                if 'heads.' not in name:
                    param.requires_grad = False
                model.heads.head.weight.data.normal_(mean=0.0, std=0.01)
                model.heads.head.bias.data.zero_()
            else:
                if 'fc.' not in name:
                    param.requires_grad = False
                model.fc.weight.data.normal_(mean=0.0, std=0.01)
                model.fc.bias.data.zero_()

    #### Load pre-trained
    if args.load_model is not None:
        print(f'### load model logged under {args.load_model}...')
        model_ckpt_dir = os.path.join(args.load_model, 'ckpt')
        assert os.path.exists(args.load_model)
        assert os.path.exists(model_ckpt_dir)
        ckpts = sorted(glob(os.path.join(model_ckpt_dir, '*.ckpt')))
        if args.load_best:
            ckpt_path = glob(os.path.join(model_ckpt_dir, '*_best.ckpt'))[0]
        else:
            ckpt_path = ckpts[-1]
        state_dict = torch.load(ckpt_path)
        if 'weight' in state_dict.keys():
            state_dict = state_dict['weight']
        print(f'### load model {ckpt_path}...')
        # pop fc weight & bias:
        target_state_dict = {}
        for key, value in state_dict.items():
            if 'module' in key:
                key = key.replace('module.', '')
            if args.load_moco:
                if 'encoder_q.0.' in key:
                    key = key.replace('encoder_q.0.', '')
                    if key == 'conv1.weight' and value.shape[1] != in_channels:
                        value = value.repeat((1, in_channels, 1, 1))
                    target_state_dict[key] = value
            else:
                if 'fc' not in key and 'heads' not in key:
                    if key == 'conv1.weight' and value.shape[1] != in_channels:
                        value = value.repeat((1, in_channels, 1, 1))
                    target_state_dict[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(target_state_dict, strict=False)
        print(f'### Missing keys: {missing_keys}\n### Unexpected keys: {unexpected_keys}')

    #### Resume
    if args.resume is not None:
        args.log = True
        args.save_model = True
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
        target_state_dict = {}
        for k, param in state_dict.items():
            k = k.replace('module.', '')
            target_state_dict[k] = param
        model.load_state_dict(target_state_dict)
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

    param_cnt = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'### Total # of trainable params: {param_cnt}')

    #### Optimizer
    eff_bsz = args.batch_size * world_size
    eff_lr = args.lr * eff_bsz / 16
    print(f'### Effective learning rate: {eff_lr:.5f} with base lr: {args.lr:.5f}')

    if args.disc_transfer:
        param_dict = disc_finetune_lr_dict(model, eff_lr, decay_rate=args.lr_decay)
    else:
        param_dict = [{'params': model.parameters()}]

    if args.adam:
        optimizer = optim.Adam(param_dict, lr=eff_lr, weight_decay=args.weight_decay)
    elif args.sgd:
        optimizer = optim.SGD(param_dict, lr=eff_lr, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    elif args.adamW:
        optimizer = optim.AdamW(param_dict, lr=eff_lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adadelta(param_dict, lr=eff_lr)

    torch.autograd.set_detect_anomaly(True)

    #### Scheduler
    if cur_ep-2 != -1:
        # init initial_lr
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', eff_lr)
    if args.cos:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=cur_ep-2)
    elif args.step:
        scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma, last_epoch=cur_ep-2)
    elif args.multi_step:
        scheduler = MultiStepLR(optimizer, [50, 75], gamma=0.1, last_epoch=cur_ep-2)
    else:
        scheduler = LambdaLR(optimizer, lambda x: x, last_epoch=cur_ep-2)

    # Create a SummaryWriter object for TensorBoard
    if rank == 0 and args.log:
        runs_dir = os.path.join(log_dir, f'runs_ep{cur_ep:0>5d}')
        os.makedirs(runs_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    if args.ddp:
        model = DDP(model, device_ids=[rank])

    best_train_auc = -1
    best_test_auc = -1
    best_val_auc = -1
    best_ep = -1
    best_val_report = None
    best_test_report = None
    for epoch in range(cur_ep, args.epochs + 1):
        cur_best = False
        print_report = False
        train_acc, train_auc = train(
            args, model, device, train_loader, scaler, optimizer, writer, epoch, train_weight, print_report,
        )
        val_acc, val_auc, val_report = test(args, model, device, val_loader, writer, epoch, print_report=print_report, split='Val')
        test_acc, test_auc, test_report = test(args, model, device, test_loader, writer, epoch, print_report=print_report, split='Test')

        if writer:
            writer.add_scalar('Epoch Acc/train', train_acc, epoch - 1)
            writer.add_scalar('Epoch Acc/test', test_acc, epoch - 1)
            writer.add_scalar('Epoch Acc/val', val_acc, epoch - 1)
            writer.add_scalar('Epoch AUC/train', train_auc, epoch - 1)
            writer.add_scalar('Epoch AUC/test', test_auc, epoch - 1)
            writer.add_scalar('Epoch AUC/val', val_auc, epoch - 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch - 1)

        # Log model gradients
        if writer and args.dev:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                if param.grad is not None:
                    writer.add_histogram(name + '/grad', param.grad, epoch)
        scheduler.step()

        best_train_auc = max(train_auc, best_train_auc)
        best_test_auc = max(test_auc, best_test_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            cur_best = True
            best_ep = epoch
            best_val_report = val_report
            best_test_report = test_report
            print(f'#### Best val report:\n{best_val_report}')
            print(f'#### Best test report:\n{best_test_report}')
            print(f'#### Best train AUC {best_train_auc:.2f}, Best val AUC {best_val_auc:.2f}, Best test AUC {best_test_auc:.2f}\n')

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
            elif args.save_best:
                ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                torch.save(state_dict, ckpt_dest)
                if cur_best:
                    best_dict = os.path.join(ckpt_dir, f"{args.model_type}_best.ckpt")
                    print(f'### Update best weight with test auc: {test_auc:.4f} at epoch {epoch}')
                    shutil.copy(ckpt_dest, best_dict)
                    best_test_auc = test_auc
            else:
                ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                torch.save(state_dict, ckpt_dest)
    print(f'\n#### End of the training!!!')
    print(f'#### Best val report at {best_ep}:\n{best_val_report}')
    print(f'#### Best test report at {best_ep}:\n{best_test_report}')
    print(f'#### Best train AUC {best_train_auc:.2f}, Best val AUC {best_val_auc:.2f}, Best test AUC {best_test_auc:.2f}\n')

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