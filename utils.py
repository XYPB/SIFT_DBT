import os
import cv2
import time
from pytz import timezone
from datetime import datetime
from PIL import Image
import seaborn as sns
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from skimage.exposure import rescale_intensity

def one_hot(x, num_classes):
    x = x.squeeze()
    out = np.zeros([x.shape[0], num_classes]).astype(int)
    out[np.arange(x.shape[0]), x] = 1
    return out

def one_hot_torch(x, num_classes, device, dtype):
    x = x.squeeze()
    out = torch.zeros([x.shape[0], num_classes]).to(dtype=torch.long)
    out[torch.arange(x.shape[0]), x] = 1
    out = out.to(device=device, dtype=dtype)
    return out

def manual_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_log_dir(args):
    est = timezone('US/Eastern')
    dt = est.localize(datetime.now())
    dt_str = dt.strftime('%Y-%m-%d-%H-%M-%S')
    task_name = 'DBT_classification'
    log_dir = os.path.join('./logs', f'{task_name}_{dt_str}_{args.model_type}_{args.exp}_train_logs')
    config_dir = os.path.join(log_dir, 'config')
    return log_dir, config_dir

def log_train_val(train_test_pair, grad_norm=None, log_dir='./log'):
    for name, (tr, te) in train_test_pair.items():
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(tr)), tr, label=f'train_{name}')
        if te is not None:
            plt.plot(np.arange(len(te)), te, label=f'test_{name}')
        dest = os.path.join(log_dir, f'{name}.png')
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close()
    
    if grad_norm is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(grad_norm)), grad_norm, label='grad_norm')
        dest = os.path.join(log_dir, 'grad_norm.png')
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close()

        plt.figure()
        plt.hist(grad_norm, 10, label='grad_norm_hist')
        percentile_90 = np.percentile(grad_norm, 90)
        dest = os.path.join(log_dir, f'grad_norm_hist_{percentile_90:.4f}.png')
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close()


def log_img(ep, batch_idx, img, label, log_dir='./logs'):
    if len(img.shape) == 2:
        dest = os.path.join(log_dir, f'ep_{ep}_b_{batch_idx}_{label}.png')
        plt.imsave(dest, img, cmap='gray')
    else:
        # save no more than 5 images to save disk quota
        for ch in range(min(5, img.shape[0])):
            dest = os.path.join(log_dir, f'ep_{ep}_b_{batch_idx}_ch_{ch}_{label}.png')
            plt.imsave(dest, img[ch], cmap='gray')

def vis_ri_kernel(ri_conv, vis_step=1, mode='2D', vmin=None, vmax=None):
    weight = ri_conv.weight.detach().cpu().numpy()
    kernel = ri_conv._make_weight_matrix(ri_conv.weight).detach().cpu().numpy()
    Cout, Cin, k = weight.shape
    kernel_size = kernel.shape[-1]

    in_channels = np.arange(0, Cin, vis_step)
    out_channels = np.arange(0, Cout, vis_step)

    if mode == '2D':
        plt.figure(figsize=(2*len(out_channels), 2*len(in_channels)))
        idx = 1
        for cin in in_channels:
            for cout in out_channels:
                ax = plt.subplot(len(in_channels), len(out_channels), idx)
                plt.title(f'K_{cin}_{cout}')
                plt.imshow(kernel[cout, cin, :, :], cmap='gray', vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.colorbar()
                idx += 1
    else:
        plt.figure(figsize=(2*len(in_channels), 2*len(out_channels)))
        x = np.arange(k)
        idx = 1
        for cin in in_channels:
            for cout in out_channels:
                ax = plt.subplot(len(in_channels), len(out_channels), idx)
                plt.title(f'K_{cin}_{cout}')
                plt.ylim((vmin, vmax))
                ax.plot(x, weight[cout, cin, :])
                idx += 1

def vis_cam(nrow, ncol, model, target_layers, input_list, test_dataset=None, img_idx=None, save_path=None):
    plt.figure(figsize=(ncol, nrow))
    plt. margins(x=0)
    for i in range(nrow*ncol):
        if input_list is not None:
            input, label = input_list[i]
        else:
            input, label = test_dataset.__getitem__(img_idx[i])
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
        input_tensor = input[None]
        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        img = input.numpy().transpose((1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(visualization)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def get_specificity_with_sensitivity(y_true, y_prob, sensitivity):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(tpr - sensitivity))
    specificity = 1 - fpr[idx]
    return specificity

def plot_auc_roc_img(y_true, y_prob, auc_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return plt.gcf()


def plot_auc_roc_img_multiclass(y_true, y_prob, num_classes):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, num_classes, normalize=False, cmap='magma'):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks, rotation=90, fontsize=12)
    plt.yticks(tick_marks, tick_marks, fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def log_images_to_writer(images, writer, global_step, args, key='train'):
    if images.shape[1] != 1 and images.shape[1] != 3:
        images = images[:, :1]
    if args.temporal_model:
        images = images[:4, :, :4] # (4, C, 4, H, W)
        _, C, _, H, W = images.shape
        images = np.swapaxes(images, 1, 2).reshape((-1, C, H, W))
    else:
        images = images[:4]
    images = np.stack([rescale_intensity(images[i], out_range=(0, 1)) for i in range(len(images))], axis=0)
    writer.add_images(f'Batch Image/{key}', images, global_step)


def disc_finetune_lr_dict(model, base_lr, decay_rate):
    param_lr_dicts = []
    cur_lr = base_lr * decay_rate
    print(f'\n### Build discriminative lr for model transfer learning...')

    named_module = [_ for _ in model.named_children()]
    for name, module in reversed(named_module):
        if name == 'encoder':
            encoder_children = [_ for _ in module.named_children()]
            for name, sub_module in reversed(encoder_children):
                if name == 'layers':
                    block_children = [_ for _ in sub_module.named_children()]
                    for name, sub_sub_module in reversed(block_children):
                        cur_lr /= decay_rate
                        print(f'#### Module name: {name}\tlr: {cur_lr:.3E}')
                        param_lr_dicts.append({'params': sub_sub_module.parameters(), 'lr': cur_lr})
                else:
                    print(f'#### Module name: {name}\tlr: {cur_lr:.3E}')
                    param_lr_dicts.append({'params': sub_module.parameters(), 'lr': cur_lr})
        elif 'layer' in name or 'fc' in name or 'head' in name or 'conv' in name or 'model' in name:
            cur_lr /= decay_rate
            print(f'#### Module name: {name}\tlr: {cur_lr:.3E}')
            param_lr_dicts.append({'params': module.parameters(), 'lr': cur_lr})
        else:
            print(f'#### Module name: {name}\tlr: {cur_lr:.3E}')
            param_lr_dicts.append({'params': module.parameters(), 'lr': cur_lr})
    return param_lr_dicts

def plot_2d_scatter(x, y, labels, s=10):
    """
    Plot 2D scatter data with color according to labels.
    
    Parameters:
    - x, y: Coordinates of the points.
    - labels: Corresponding labels for each point.
    """

    # Create a scatter plot
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(x, y, c=labels, cmap='tab20', s=s,
                          edgecolors='k')
    
    # Add a colorbar with label ticks
    cbar = plt.colorbar(scatter)
    unique_labels = np.unique(labels)
    cbar.set_ticks(unique_labels)
    cbar.set_label('Labels')

    # Display the plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot colored by labels')
    plt.grid(True)
    return plt.gcf()


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def convert_seconds(seconds):
    # get hours
    hours, remainder = divmod(seconds, 3600)
    # get minutes and seconds
    minutes, seconds = divmod(remainder, 60)
    
    # format the output string as HH:MM:SS
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def vis_attn(images: np.ndarray, attn: np.ndarray, alpha: float = 0.6, 
             vis_index: int = None, vis_dir: str = './tmp', global_idx: int = 0,
             return_img: bool = False):
    N, C, H, W = images.shape
    if C == 1:
        images = np.repeat(images, 3, axis=1)
    images = np.stack([rescale_intensity(images[i], out_range=(0, 1)) for i in range(len(images))], axis=0)

    _, num_heads, S = attn.shape
    attn_size = int(S**0.5)
    attn = attn.reshape((N, num_heads, attn_size, attn_size))
    # average over heads
    attn = np.mean(attn, axis=1)
    vis_index = N // 2 if vis_index == None else vis_index

    img = np.transpose(images[vis_index], axes=[1, 2, 0])
    attn_resize = cv2.resize(attn[vis_index], (W, H), interpolation=cv2.INTER_NEAREST)
    attn_resize = rescale_intensity(attn_resize, out_range=(0, 1))
    attn_resize = plt.get_cmap('magma')(attn_resize)
    attn_resize = np.delete(attn_resize, 3, 2)
    overlay = alpha * img +  (1 - alpha) * attn_resize
    overlay = rescale_intensity(overlay, out_range=(0, 1))
    if return_img:
        return attn_resize, overlay
    attn_dest = os.path.join(vis_dir, f'{global_idx}_{vis_index}_attn.png')
    overlay_dest = os.path.join(vis_dir, f'{global_idx}_{vis_index}_overlay.png')
    plt.imsave(attn_dest, attn_resize)
    plt.imsave(overlay_dest, overlay)

def calc_model_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        # EMA model norm ignored
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def plot_logits_distribution(output, im_dest=None):
    plt.cla()
    plt.close('all')
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Logits')
    plt.hist(output, 20)
    plt.subplot(1, 2, 2)
    plt.title('Slice-wise logits')
    plt.bar(np.arange(output.shape[0]), output, 1.0)
    plt.tight_layout()
    if im_dest != None:
        plt.savefig(im_dest, dpi=300)
    else:
        return plt.gcf()