from typing import Iterable
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import random
import cv2

def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def patchify_2D_array(array, h, w):
    height, width = array.shape
    patches = []
    
    for i in range(0, height, h):
        for j in range(0, width, w):
            patch = array[i:i+h, j:j+w]
            patches.append(patch)
    
    return np.array(patches)


class Padding(object):

    def __init__(self, target_size, return_array=False) -> None:
        self.target_size = target_size
        self.return_array = return_array

    def __process__(self, x):
        W, H = x.size
        assert H <= self.target_size[0] and W <= self.target_size[1]
        pad_h_after = int((self.target_size[0] - H) // 2)
        pad_h_before = self.target_size[0] - H - pad_h_after
        pad_w_after = int((self.target_size[1] - W) // 2)
        pad_w_before = self.target_size[1] - W - pad_w_after
        x = np.array(x)
        out = np.pad(x, [[pad_h_after, pad_h_before], [pad_w_before, pad_w_after]])
        if self.return_array:
            return out
        else:
            return Image.fromarray(out)

    def __call__(self, x):
        if isinstance(x, Iterable):
            out = [self.__process__(im) for im in x]
            if self.return_array:
                return np.stack(out, axis=-1)
            else:
                return out
        else:
            return self.__process__(x)


class ResizeLongSide(object):

    def __init__(self, target_size) -> None:
        self.target_size = target_size

    def __process__(self, x):
        W, H = x.size
        if H > W:
            h = self.target_size[0]
            scale = h / H
            w = int(W * scale)
            return x.resize((w, h))
        else:
            w = self.target_size[1]
            scale = w / W
            h = int(H * scale)
            return x.resize((w, h))

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)


class Rescale(object):
    
    def __init__(self) -> None:
        pass

    def __process__(self, x):
        min_val, max_val = x.min(), x.max()
        return (x - min_val) / (max_val - min_val)

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)


class OtsuCut(object):

    def __init__(self):
        super().__init__()

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        mask = otsu_mask(x)
        col_sum = np.sum(mask, axis=0)
        row_sum = np.sum(mask, axis=1)
        x1x2 = np.argwhere(col_sum).squeeze()
        y1y2 = np.argwhere(row_sum).squeeze()
        x = x[y1y2[0]:y1y2[-1], x1x2[0]:x1x2[-1]]
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)



class Patchify(object):

    def __init__(self, patch_size, patch_cnt, eps=1e-2, return_array=False):
        self.patch_size = patch_size
        self.patch_cnt = patch_cnt
        self.eps = eps
        self.return_array = return_array

    def __process__(self, im):
        if isinstance(im, Image.Image):
            im = np.array(im)
        mask = otsu_mask(im)
        patches = patchify_2D_array(im, self.patch_size[0], self.patch_size[1])
        masks = patchify_2D_array(mask, self.patch_size[0], self.patch_size[1])
        mean_masks = np.mean(masks, axis=(1, 2))
        non_zero_idx = (mean_masks > self.eps)
        patches = patches[non_zero_idx]
        num_patches = patches.shape[0]
        assert self.patch_cnt <= num_patches
        sampled_idx = random.sample(list(range(num_patches)), k=self.patch_cnt)
        patches = patches[np.array(sampled_idx).astype(int)]
        return patches

    def __call__(self, x):
        if isinstance(x, Iterable):
            out = [self.__process__(im) for im in x]
            if self.return_array:
                return np.concatenate(out, axis=0).transpose((1, 2, 0))
            else:
                return out
        else:
            return self.__process__(x)


def get_transforms(args):

    transform = [ResizeLongSide((args.target_H, args.target_W)),]
    test_transform = [ResizeLongSide((args.target_H, args.target_W)),]

    if args.patchify:
        transform += [
            Padding((args.target_H, args.target_W), return_array=False),
            Patchify((args.patch_size, args.patch_size), args.patch_cnt, args.patch_eps, return_array=True),
            transforms.ToTensor(),
        ]
        test_transform += [
            Padding((args.target_H, args.target_W), return_array=False),
            Patchify((args.patch_size, args.patch_size), args.patch_cnt, args.patch_eps, return_array=True),
            transforms.ToTensor(),
        ]
    else:
        transform += [
            Padding((args.target_H, args.target_W), return_array=True),
            transforms.ToTensor(),
        ]
        test_transform += [
            Padding((args.target_H, args.target_W), return_array=True),
            transforms.ToTensor(),
            transforms.Normalize((0.1550), (0.1521)),
        ]

    if args.moco_aug:
        transform += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((args.target_H, args.target_W), scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 1.0])], p=0.2),
            transforms.RandomApply(
                [transforms.RandomAffine(args.rotate_degree, (args.translate_ratio, args.translate_ratio))],
                p=args.affine_prob,
            ),
            transforms.Normalize((0.1550), (0.1521)),
        ]
    else:
        transform += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(args.rotate_degree, (args.translate_ratio, args.translate_ratio))],
                p=args.affine_prob,
            ),
            transforms.Normalize((0.1550), (0.1521)),
        ]

    transform = transforms.Compose(transform)
    test_transform = transforms.Compose(test_transform)
    return transform, test_transform


