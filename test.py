import os
from glob import glob

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import torchvision
from lightly.models.modules import MoCoProjectionHead
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from opt import get_opt
from dataset.BCS_DBT_cls_dataset import get_dataset
from dataset.transforms import get_transforms
from utils import (manual_seed, one_hot, plot_auc_roc_img, 
                   vis_attn, get_specificity_with_sensitivity,
                   plot_confusion_matrix, plot_logits_distribution)
from models.simple_cnn import SimpleCNN


def test(args, model, device, test_loader, test_dir, print_report=False):
    model.eval()
    outputs = []
    max_outputs = []
    targets = []
    vol2pred = {}
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for idx, (data, target, views, paths, dcm_paths) in pbar:
            if args.dev and idx % 100 != 0:
                continue
            data, target = data.to(device), target.to(device)
            if len(target.shape) > 1:
                target = target.squeeze(dim=1)
            if args.abnormal_only:
                if target.item() == 0:
                    continue
            if not args.patch_lv:
                # only test for one volume at a time
                assert data.shape[0] == 1 and target.shape[0] == 1
                data = torch.transpose(data, 0, 1) #(num_slice, 1, H, W)
            if args.patch_cnt > 1:
                assert data.shape[1] == args.patch_cnt
                N, num_patch, H, W = data.shape
                data = torch.reshape(data, (N*num_patch, 1, H, W))
                target = target[:, 0] # same label over patches from the same volume

            dtype = torch.bfloat16 if args.bf16 else None
            with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
                output = model(data)
                output = F.softmax(output, dim=1)
            if args.bf16:
                output = output.float()
            if not args.patch_lv:
                # maximize over all slices
                if args.plot_logits:
                    plot_logits_distribution(output[:, 1], im_dest=os.path.join(test_dir, f'pred_logits_distribution_{idx}_{target.item()}.png'))
                max_positive_idx = torch.argmax(output[:, 1].detach(), dim=0)
                max_output = output[max_positive_idx:max_positive_idx+1]
                max_outputs.append(max_output.detach().cpu().numpy())
            if args.patch_cnt > 1:
                output = torch.reshape(output, (N, num_patch, -1))
                if args.max_patch:
                    max_positive_idx = torch.argmax(output[:, :, 1].detach(), dim=1)
                    output = output[torch.arange(N), max_positive_idx]
                else:
                    output = torch.mean(output, dim=1)
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            outputs.append(output)
            targets.append(target)
            for i, dcm_path in enumerate(dcm_paths):
                if dcm_path not in vol2pred.keys():
                    vol2pred[dcm_path] = []
                vol2pred[dcm_path].append((output[i], target[i]))

            if args.dry_run and idx == 2:
                break

    if not args.patch_lv:
        outputs = max_outputs
    outputs = np.concatenate(outputs, axis=0)
    preds = np.argmax(outputs, axis=1).squeeze()
    targets = np.concatenate(targets, axis=0).squeeze()
    correct = np.sum(targets == preds)
    acc = 100 * accuracy_score(targets, preds)
    if args.binary:
        auc = 100 * roc_auc_score(targets, outputs[:, 1])
        specificity_087 = 100 * get_specificity_with_sensitivity(targets, outputs[:, 1], 0.87)
        specificity_080 = 100 * get_specificity_with_sensitivity(targets, outputs[:, 1], 0.80)
    else:
        auc = 100 * roc_auc_score(one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class='ovr')
        specificity_087 = specificity_080 = 0.0

    if os.path.exists(test_dir):
        prefix = 'train' if args.train_data else 'test'
        curve_path = os.path.join(test_dir, f'{prefix}_auc_roc_curve_{auc:.2f}_{specificity_087:.2f}_{specificity_080:.2f}.png')
        auc_roc = plot_auc_roc_img(targets, outputs[:, 1], auc)
        auc_roc.savefig(curve_path, dpi=300, bbox_inches='tight')
        confusion_path = os.path.join(test_dir, f'{prefix}_confusion_mat.png')
        confusion_mat = plot_confusion_matrix(targets, preds, num_classes=outputs.shape[1], normalize=True)
        confusion_mat.savefig(confusion_path, dpi=300, bbox_inches='tight')
        output_path = os.path.join(test_dir, f'{prefix}_output')
        np.savez_compressed(output_path, vol2pred=vol2pred)

    print(f'\nTest set: Accuracy: {correct}/{len(targets)} ({acc:.2f}%), AUC: {auc:.2f}%, Specificity at 87%: {specificity_087:.2f}, Specificity at 80%: {specificity_080:.2f}\n')
    if print_report:
        print(classification_report(targets, preds, digits=4, zero_division=0.0))
    return acc, auc, (specificity_087, specificity_080)


def extract_feat(args, model, device, test_loader):
    model.eval()
    feats = []
    targets = []
    overall_dcm_paths = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    if args.extract_attn:
        attn_dir = os.path.join(args.load_model, 'attn_vis')
        os.makedirs(attn_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (data, target, views, paths, dcm_paths) in pbar:
            data, target = data.to(device), target.to(device)
            if len(target.shape) > 1:
                target = target.squeeze(dim=1)
            if args.abnormal_only:
                if target.item() == 0:
                    continue
            if not args.patch_lv:
                # only test for one patient at a time
                assert data.shape[0] == 1 and target.shape[0] == 1
                data = torch.transpose(data, 0, 1) #(num_slice, 1, H, W)
            if args.patch_cnt > 1:
                assert data.shape[1] == args.patch_cnt
                N, num_patch, H, W = data.shape
                data = torch.reshape(data, (N*num_patch, 1, H, W))
                target = target[:, 0] # same label over patches from the same volume
            dtype = torch.bfloat16 if args.bf16 else None
            with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
                if args.vqvae:
                    if args.divide:
                        split = data.shape[0] // 2
                        output1 = model.encode_pool(data[:split])
                        output2 = model.encode_pool(data[split:])
                        output = torch.cat([output1, output2], dim=0)
                    else:
                        output = model.encode_pool(data)
                else:
                    if args.extract_attn:
                        output = model[0].get_last_blk_attn(data)
                        # only take [CLS] attention map
                        output = output[:, :, 0, 1:]
                        vis_attn(
                            data.detach().cpu().numpy(), output.detach().cpu().numpy(), 
                            vis_index=(output.shape[0]//2), vis_dir=attn_dir, 
                            global_idx=idx
                        )
                    else:
                        output = model(data)
            if args.bf16:
                output = output.float()
            if args.patch_cnt > 1:
                output = torch.reshape(output, (N, num_patch, -1))
                if args.max_patch:
                    max_positive_idx = torch.argmax(output[:, :, 1].detach(), dim=1)
                    output = output[torch.arange(N), max_positive_idx]
                else:
                    output = torch.mean(output, dim=1)
            if idx == 0 and args.dev:
                print(output.shape)


            feats.append(output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            for dcm_path in dcm_paths:
                overall_dcm_paths.append(dcm_path)
            if idx == 2 and args.dry_run:
                break

    if args.patch_lv:
        feats = np.concatenate(feats, axis=0) # (N, C)
    targets = np.concatenate(targets, axis=0).squeeze()

    assert os.path.exists(args.load_model)
    prefix = 'train' if args.train_data else 'test'
    if args.ignore_action:
        feat_path = os.path.join(args.load_model, f'{prefix}_extracted_feats.npz')
    elif args.extract_attn:
        feat_path = os.path.join(args.load_model, f'{prefix}_extracted_attn_map.npz')
    else:
        feat_path = os.path.join(args.load_model, f'{prefix}_extracted_feats_w_action.npz')
    if args.patch_cnt > 1:
        feat_path = feat_path.replace('.npz', f'_patch_cnt_{args.patch_cnt}.npz')
    assert len(feats) == len(targets) == len(overall_dcm_paths)
    np.savez_compressed(feat_path, {'feats': feats, 'targets': targets, 'dcm_paths': overall_dcm_paths})
    return


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        if args.num_workers == -1:
            args.num_workers = len(os.sched_getaffinity(0))
        print(f'### Use {args.num_workers} cores for training...')
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    if args.cudnn:
        cudnn.benchmark = True

    _, test_transform = get_transforms(args)
    if args.train_data:
        test_dataset = get_dataset('data/csv/BCS-DBT_train_label_v2.csv', test_transform, args, test=True, get_info=True)
    else:
        test_dataset = get_dataset('data/csv/BCS-DBT_test_label_v2.csv', test_transform, args, test=True, get_info=True)


    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if args.binary:
        n_classes = 2
    elif args.vindr:
        n_classes = 5
    else:
        n_classes = 4

    ### In-channels
    in_channels = args.num_slice
    if args.patchify:
        in_channels *= args.patch_cnt

    #### Model
    model_args = {'num_classes': n_classes}
    if 'simple' in args.model_type:
        model = SimpleCNN(in_channels, model_args['num_classes'], base_channels=32, dropout_p=args.dropout_p)
        in_features = model.fc.in_features
    else:
        if 'resnet' in args.model_type:
            model = getattr(torchvision.models, args.model_type)(**model_args)
            in_features = model.fc.in_features
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise NotImplementedError

    if args.load_moco and args.extract_feat:
        model.fc = nn.Identity()
        model = nn.Sequential(
            model,
            MoCoProjectionHead(input_dim=in_features, hidden_dim=in_features, output_dim=128)
        )
    elif args.extract_feat:
        if 'resnet' in args.model_type:
            model.fc = nn.Identity()
        elif 'vit' in args.model_type:
            model.heads.head = nn.Identity()
        elif 'simple' in args.model_type:
            model.fc = nn.Identity()

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
            epoch = state_dict['epoch']
            state_dict = state_dict['weight']
        else:
            epoch = -1
        print(f'### Load model from {ckpt_path} at epoch {epoch}...')
        target_state_dict = {}
        for key, value in state_dict.items():
            if 'module.' in key:
                key = key.replace('module.', '')
            if args.load_moco:
                if 'encoder_q.' in key:
                    key = key.replace('encoder_q.', '')
                    target_state_dict[key] = value
            else:
                target_state_dict[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(target_state_dict, strict=False)
        print(f'### Missing keys: {missing_keys}\n### Unexpected keys: {unexpected_keys}')
    else:
        raise NotImplementedError

    test_cnt = len(glob(os.path.join(args.load_model, 'test_dir*')))
    cur_test_dir = os.path.join(args.load_model, f'test_dir_{str(test_cnt).zfill(4)}')
    os.makedirs(cur_test_dir, exist_ok=False)
    print(f'### Save test results under {cur_test_dir}...')

    if args.extract_feat:
        extract_feat(args, model, device, test_loader)
    else:
        test(args, model, device, test_loader, cur_test_dir, print_report=True)

if __name__ == '__main__':
    args = get_opt()
    main(args)