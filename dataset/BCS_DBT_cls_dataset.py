import torch
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
from collections import Counter 
from glob import glob
import json
import torchvision.transforms as transforms
import random
from dataset.duke_dbt_data import dcmread_image
from .transforms import Padding, ResizeLongSide, Rescale, otsu_mask
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from .utils import calc_iou, calc_intersection, calc_box


def get_other_view(view):
    side = view[0]
    if 'cc' in view:
        return side + 'mlo'
    else:
        return side + 'cc'

class BCS_DBT_cls(torch.utils.data.Dataset):

    def __init__(self, csv_path, load_from_raw=False, use_otsu=False, transform=None, binary=False, 
                 subset=False, subset_ratio=1.0, mid_slice=False, ignore_action=False, num_slice=1, 
                 sampling_gap=None, fix_gap=False, uniform_norm=False, contrastive=False,
                 df_box_path='data/csv/BCS-DBT boxes-train-v2.csv', pick_mass_slice=False, 
                 load_from_npz=False, temporal=False, positive_range=4, inter_slice=False,
                 dino=False, inter_view=False, inter_study=False, inter_patient=False, 
                 ignore_abnormal=False, volume_orig_size='data/BCS-DBT_volume_orig_size.json',
                 get_info=False):
        super().__init__()
        self.split = csv_path.split('/')[-1].split('_')[1]
        self.df = pd.read_csv(csv_path)
        self.df_box = pd.read_csv(df_box_path)
        self.volume_orig_size = json.load(open(volume_orig_size, 'r'))
        self.load_from_raw = load_from_raw
        self.use_otsu = use_otsu
        self.binary = binary
        self.transform = transform
        self.subset = subset
        self.subset_ratio = subset_ratio
        self.mid_slice = mid_slice
        self.ignore_action = ignore_action
        self.ignore_abnormal = ignore_abnormal
        self.num_slice = num_slice
        self.sampling_gap = sampling_gap
        self.fix_gap = fix_gap
        self.uniform_norm = uniform_norm
        self.pick_mass_slice = pick_mass_slice
        self.contrastive = contrastive
        self.load_from_npz = load_from_npz
        self.temporal = temporal
        self.positive_range = positive_range
        self.inter_slice = inter_slice
        self.dino = dino
        self.inter_view = inter_view
        self.inter_study = inter_study
        self.inter_patient = inter_patient
        self.get_info = get_info
        if self.inter_view or self.inter_study or self.inter_patient:
            self.inter_slice = True

        self.slice_cnt = json.load(open('data/BCS-DBT_slice_cnt.json', 'r'))
        self.__make_labels__()
        self.__make_train_idx__()
        self.__process_box__()
        self.__process_view__()

    def __process_box__(self):
        idx2box_info = {}
        for idx in range(len(self.train_idx)):
            df_idx = self.train_idx[idx]
            entry = self.df.iloc[df_idx]
            if entry['Normal'] or entry['Actionable']:
                continue
            sid = entry['StudyUID']
            view = entry['View']
            box = self.df_box[self.df_box['StudyUID'] == sid]
            box = box[box['View'] == view]
            orig_size = [self.volume_orig_size[str(b_idx)] for b_idx in box.index]
            b_info = box.to_dict('list')
            b_info['OrigSize'] = orig_size
            idx2box_info[idx] = b_info
        self.idx2box_info = idx2box_info

    def __process_view__(self):
        idx2patient_idx = {}
        idx2study_idx = {}
        idx2view_idx = {}
        for idx in range(len(self.train_idx)):
            df_idx = self.train_idx[idx]
            entry = self.df.iloc[df_idx]
            pid = entry['PatientID']
            sid = entry['StudyUID']
            view = entry['View']
            other_view = get_other_view(view)
            same_patient_idx = self.df.loc[self.df['PatientID'] == pid].index.tolist()
            same_patient_idx.remove(df_idx)
            idx2patient_idx[idx] = same_patient_idx

            same_study_idx = self.df.loc[self.df['StudyUID'] == sid].index.tolist()
            same_study_idx.remove(df_idx)
            idx2study_idx[idx] = same_study_idx

            other_view_df_idx = self.df.loc[(self.df['StudyUID'] == sid)\
                                             & (self.df['View'] == other_view)].index.tolist()
            idx2view_idx[idx] = other_view_df_idx

        self.idx2patient_idx = idx2patient_idx
        self.idx2study_idx = idx2study_idx
        self.idx2view_idx = idx2view_idx


    def __len__(self):
        return len(self.train_idx)

    def __load_from_dcm__(self, dcm_path, slice_idx, view, png_folder):
        image = dcmread_image(dcm_path, view=view)
        image = (image // 257).astype(np.uint8)
        img = [Image.fromarray(image[i]) for i in slice_idx]

        # fix missing image
        for i, idx in enumerate(slice_idx):
            img_path = os.path.join(png_folder, f'slice_{str(idx).zfill(4)}.png')
            img[i].save(img_path)

        return img
    
    def __load_from_npz__(self, dcm_path, slice_idx):
        if self.use_otsu:
            npz_path = dcm_path.replace('.dcm', '_otsu.npz')
        else:
            npz_path = dcm_path.replace('.dcm', '.npz')
        image_stack = np.load(npz_path, allow_pickle=True)['arr']
        imgs = [image_stack[i] for i in slice_idx]
        if self.uniform_norm:
            imgs = [rescale_intensity(img, out_range=(0, 1)) for img in imgs]
        return [Image.fromarray(img) for img in imgs], npz_path

    def __make_train_idx__(self):
        idx = []
        labels = []
        if self.subset:
            if self.ignore_abnormal:
                subset_config_path = f'./data/BCS-DBT_{self.split}_subset_idx_{self.subset_ratio:.2f}_normal.json'
            elif self.ignore_action:
                subset_config_path = f'./data/BCS-DBT_{self.split}_subset_idx_{self.subset_ratio:.2f}.json'
            else:
                subset_config_path = f'./data/BCS-DBT_{self.split}_subset_idx_{self.subset_ratio:.2f}_w_action.json'
            print(f'### Subset split file stored in {subset_config_path}...')
            if os.path.exists(subset_config_path):
                with open(subset_config_path, 'r') as fp:
                    idx, labels = json.load(fp)
            else:
                label_cnt = Counter(self.labels)
                label_prob = {lb: 1-(cnt/len(self.df)) for lb, cnt in label_cnt.items()}
                for i, label in enumerate(self.labels):
                    if label == 0:
                        if np.random.uniform(0, 1 / self.subset_ratio) < label_prob[label]:
                            idx.append(i)
                            labels.append(label)
                    elif self.ignore_action and label == 1:
                        continue
                    elif self.ignore_abnormal and label > 0:
                        continue
                    else:
                        idx.append(i)
                        labels.append(label)
                with open(subset_config_path, 'w') as fp:
                    json.dump((idx, labels), fp)
        else:
            for i, label in enumerate(self.labels):
                if self.ignore_action and label == 1:
                    continue
                if self.ignore_abnormal and label > 0:
                    continue
                else:
                    idx.append(i)
                    labels.append(label)
        print('### Sampled subset distribution: ', Counter(labels))
        self.train_idx = idx

    def __make_labels__(self):
        labels = []
        for i in range(len(self.df)):
            entry = self.df.iloc[i]
            if int(entry['Cancer']):
                label = 3
            elif int(entry['Benign']):
                label = 2
            elif int(entry['Actionable']):
                label = 1
            elif int(entry['Normal']):
                label = 0
            else:
                raise NotImplementedError
            labels.append(label)
        self.labels = labels

    def __getitem_help__(self, df_idx, box_info, get_info=False, slice_idx=None):
        entry = self.df.iloc[df_idx]
        label = self.labels[df_idx]
        if self.binary and label > 0:
            label = 1
        dcm_path = os.path.join('./data', entry['classic_path'])
        view = entry['View']
        png_folder = dcm_path.replace('1-1.dcm', 'png')
        slice_cnt = self.slice_cnt[entry['classic_path']]
        num_slice = slice_cnt if self.num_slice == 'all' else self.num_slice

        if self.load_from_raw:
            dcm_path = os.path.join('./data', entry['classic_path'])
            slice_idx = random.sample(list(range(slice_cnt)), k=num_slice)
            img = self.__load_from_dcm__(dcm_path, slice_idx, view, png_folder)
        else:
            # sample slices from the z-stack
            if slice_idx is not None:
                slice_idx = slice_idx
            elif self.pick_mass_slice and (entry['Cancer'] or entry['Benign']):
                pos_idx = []
                mass_slice_idx = box_info['Slice']
                for ms_idx in mass_slice_idx:
                    pos_idx += list(range(max(0, ms_idx - self.positive_range), 
                                    min(slice_cnt, ms_idx + self.positive_range)))
                pos_idx = list(set(pos_idx))
                mass_slice_idx = random.sample(pos_idx, k=1)[0]
                min_slice_idx = max(0, mass_slice_idx - (num_slice // 2))
                if min_slice_idx + num_slice > slice_cnt:
                    min_slice_idx = slice_cnt - num_slice
                slice_idx = [min_slice_idx + _i for _i in range(num_slice)]
                assert mass_slice_idx in slice_idx
            elif self.mid_slice:
                slice_idx = [int(slice_cnt // 2)]
            elif self.fix_gap:
                sampling_gap = int(slice_cnt // (num_slice + 1))
                slice_idx = random.sample(list(range(slice_cnt - ((num_slice - 1) * sampling_gap))), k=1)[0]
                slice_idx = [slice_idx + sampling_gap * i for i in range(num_slice)]
                assert len(slice_idx) == num_slice
            elif self.sampling_gap is not None:
                slice_idx = random.sample(list(range(slice_cnt - ((num_slice - 1) * self.sampling_gap))), k=1)[0]
                slice_idx = [slice_idx + self.sampling_gap * i for i in range(num_slice)]
            elif self.inter_slice:
                slice_idx = random.sample(list(range(slice_cnt)), k=1)[0]
                # Okay to sample two same slices
                second_slice_range = list(range(max(0, slice_idx - self.positive_range), min(slice_cnt, slice_idx + self.positive_range)))
                slice_idx = [slice_idx] + random.sample(second_slice_range, k=1)
            else:
                slice_idx = random.sample(list(range(slice_cnt)), k=num_slice)

            #### Load images
            try:
                if self.load_from_npz:
                    img, image_paths = self.__load_from_npz__(dcm_path, slice_idx)
                else:
                    img = []
                    image_paths = []
                    for i in slice_idx:
                        if self.use_otsu:
                            img_path = os.path.join(png_folder, f'slice_{str(i).zfill(4)}_otsu.png')
                        else:
                            img_path = os.path.join(png_folder, f'slice_{str(i).zfill(4)}.png')
                        cur_img = Image.open(img_path)
                        if self.uniform_norm:
                            cur_img = rescale_intensity(np.array(cur_img), out_range=(0, 1))
                            cur_img = Image.fromarray(cur_img)
                        img.append(cur_img)
                        image_paths.append(img_path)
            except UnidentifiedImageError as e:
                print(f'!!! Failed on loading image: {img_path} due to {e}, load from npz')
                img, image_paths = self.__load_from_npz__(dcm_path, slice_idx)
            except FileNotFoundError as e:
                print(f'!!! Failed on loading image: {img_path} due to {e}, load from npz')
                img, image_paths = self.__load_from_npz__(dcm_path, slice_idx)
            except Exception as e:
                raise e

        if self.transform is not None:
            if self.inter_slice:
                if self.dino:
                    views1 = self.transform(img[0])
                    views2 = self.transform(img[1])
                    n_local = len(views1) - 2
                    # pick global view & local views from image 1 and 2
                    img_t = [views1[0], views2[0]] + views1[2:2+n_local//2] + views1[2+n_local//2:]
                else:
                    key, _ = self.transform(img[0])
                    query, _ = self.transform(img[1])
                    img_t = (key, query)
            else:
                if self.contrastive:
                    img = img[0]
                img_t = self.transform(img)

        if self.temporal:
            # (N, T, H, W) -> (N, 1, T, H, W)
            img_t = torch.unsqueeze(img_t, dim=0)

        if get_info:
            return img_t, label, view, image_paths
        else:
            return img_t, label
        
    def __getitem__(self, idx, get_info_flag=False, slice_idx=None):
        df_idx = self.train_idx[idx]
        box_info = self.idx2box_info.get(idx, None)
        get_info = self.get_info or get_info_flag
        if get_info:
            return self.__getitem_help__(df_idx, box_info, get_info, slice_idx)
        else:
            img_t_1, label_1 = self.__getitem_help__(df_idx, box_info, False, slice_idx)
        if self.inter_view:
            other_idx_list = self.idx2view_idx[idx]
        elif self.inter_patient:
            other_idx_list = self.idx2patient_idx[idx]
        elif self.inter_study:
            other_idx_list = self.idx2study_idx[idx]
        else:
            return img_t_1, label_1
        if len(other_idx_list) == 0:
            return img_t_1, label_1
        other_idx = random.sample(other_idx_list, k=1)[0]
        other_box_info = self.idx2box_info.get(other_idx, None)
        # 50% chance to use inter-positive
        if random.uniform(0, 1) > 0.5:
            return img_t_1, label_1

        entry1 = self.df.iloc[df_idx]
        view1 = entry1['View']
        study1 = entry1['StudyUID']
        p1 = entry1['PatientID']
        entry2 = self.df.iloc[other_idx]
        view2 = entry2['View']
        study2 = entry2['StudyUID']
        p2 = entry2['PatientID']
        if self.inter_view:
            assert p1 == p2
            assert study1 == study2
            assert view1 != view2
        elif self.inter_patient:
            assert p1 == p2
        elif self.inter_study:
            assert p1 == p2
            assert study1 == study2


        img_t_2, label_2 = self.__getitem_help__(other_idx, other_box_info, False, None)
        if self.dino:
            n_local = len(img_t_2) - 2
            img_t = [img_t_1[0], img_t_2[0]] + img_t_1[2:2+n_local//2] + img_t_2[2+n_local//2:]
        else:
            img_t = (img_t_1[0], img_t_2[0])
        label_t = label_1 if label_1 != 0 else label_2
        return img_t, label_t


class BCS_DBT_patch_cls(BCS_DBT_cls):

    def __init__(self, csv_path, load_from_raw=False, use_otsu=False, transform=None, binary=False, subset=False,
                 subset_ratio=1.0, mid_slice=False, ignore_action=False, num_slice=1, sampling_gap=None, 
                 fix_gap=False, uniform_norm=False, df_box_path='data/csv/BCS-DBT boxes-train-v2.csv', 
                 contrastive=False, pick_mass_slice=False, patch_size=224, positive_range=4, 
                 patch_eps=0.2, mass_eps=0.2, load_from_npz=False, temporal=False, ignore_abnormal=False,
                 patch_cnt=1, get_info=False):
        super().__init__(csv_path, load_from_raw=load_from_raw, use_otsu=use_otsu, transform=transform, 
                         binary=binary, subset=subset, subset_ratio=subset_ratio, mid_slice=mid_slice,
                         ignore_action=ignore_action, num_slice=num_slice, sampling_gap=sampling_gap, 
                         fix_gap=fix_gap, uniform_norm=uniform_norm, df_box_path=df_box_path,
                         pick_mass_slice=pick_mass_slice, contrastive=contrastive, 
                         load_from_npz=load_from_npz, temporal=temporal, positive_range=positive_range,
                         ignore_abnormal=ignore_abnormal, get_info=get_info)
        self.patch_size = patch_size
        self.patch_eps = patch_eps
        self.mass_eps = mass_eps
        self.patch_cnt = patch_cnt

        self.__make_patch_level_dataset__()

    def __make_patch_level_dataset__(self):
        patch_idx = []
        patch_labels = []
        patch_box_info = {}
        for idx, df_idx in enumerate(self.train_idx):
            label = self.labels[df_idx]
            entry = self.df.iloc[df_idx]
            slice_cnt = self.slice_cnt[entry['classic_path']]
            patch_idx += [(df_idx, i) for i in range(slice_cnt)]
            pos_idx = []
            if entry['Cancer'] or entry['Benign']:
                box_info = self.idx2box_info[idx]
                patch_box_info[df_idx] = box_info
                mass_slice_idx = box_info['Slice']
                for ms_idx in mass_slice_idx:
                    pos_idx += list(range(max(0, ms_idx - self.positive_range), 
                                          min(slice_cnt, ms_idx + self.positive_range)))
            pos_idx = set(pos_idx)
            patch_labels += [0 if i not in pos_idx else label for i in range(slice_cnt)]
            
        assert len(patch_idx) == len(patch_labels)
        print('### Sampled subset patch level distribution: ', Counter(patch_labels))
        self.train_idx = patch_idx
        self.labels = patch_labels
        self.idx2box_info = patch_box_info

    def __sample_patch__(self, img, mask):
        H, W = img.shape
        while True:
            h1, w1 = random.randint(0, H - self.patch_size), random.randint(0, W - self.patch_size)
            mask_mean = np.mean(mask[h1:h1+self.patch_size, w1:w1+self.patch_size]) / np.max(mask)
            if mask_mean > self.patch_eps:
                break
        return h1, w1, h1+self.patch_size, w1+self.patch_size


    def __getitem__(self, idx, get_info_flag=False, slice_idx=None):
        get_info = self.get_info or get_info_flag
        df_idx, slice_idx = self.train_idx[idx]
        label = self.labels[idx]
        entry = self.df.iloc[df_idx]
        if self.binary and label > 0:
            label = 1
        dcm_path = os.path.join('./data', entry['classic_path'])
        view = entry['View']
        png_folder = dcm_path.replace('1-1.dcm', 'png')

        slice_cnt = self.slice_cnt[entry['classic_path']]
        num_slice = slice_cnt if self.num_slice == 'all' else self.num_slice

        img = []
        image_paths = []
        labels = []
        if self.load_from_npz:
            cur_img, img_path = self.__load_from_npz__(dcm_path, [slice_idx])
            cur_img = np.array(cur_img[0])
        else:
            if self.use_otsu:
                img_path = os.path.join(png_folder, f'slice_{str(slice_idx).zfill(4)}_otsu.png')
            else:
                img_path = os.path.join(png_folder, f'slice_{str(slice_idx).zfill(4)}.png')
            try:
                cur_img = np.array(Image.open(img_path))
                H, W = cur_img.shape
            except UnidentifiedImageError as e:
                print(f'!!! Failed on loading image: {img_path} due to {e}, load from npz')
                cur_img, img_path = self.__load_from_npz__(dcm_path, [slice_idx])
                cur_img = np.array(cur_img[0])
            except FileNotFoundError as e:
                print(f'!!! Failed on loading image: {img_path} due to {e}, load from npz')
                cur_img, img_path = self.__load_from_npz__(dcm_path, [slice_idx])
                cur_img = np.array(cur_img[0])
            except Exception as e:
                raise e
        H, W = cur_img.shape
        mask = otsu_mask(cur_img)

        for patch_idx in range(self.patch_cnt):
            if (entry['Cancer'] or entry['Benign']) and label > 0:
                box_info = self.idx2box_info[df_idx]
                # find closest mass slice among multiple slices
                mass_slice_idx = box_info['Slice']
                mass_box_idx = np.argmin([abs(ii - slice_idx) for ii in mass_slice_idx])
                # ensure patch including mass
                x1, y1 = box_info['X'][mass_box_idx], box_info['Y'][mass_box_idx]
                width, height = box_info['Width'][mass_box_idx], box_info['Height'][mass_box_idx]
                orig_H, orig_W = box_info['OrigSize'][mass_box_idx]
                ratio = float(H) / float(orig_H)
                x1, y1, width, height = [round(v * ratio) for v in (x1, y1, width, height)]
                box_mass = (y1, x1, y1 + height, x1 + width)

                min_poss_h = max(0, box_mass[2] - self.patch_size)
                max_poss_h = min(H - self.patch_size,  box_mass[0])
                min_poss_w = max(0, box_mass[3] - self.patch_size)
                max_poss_w = min(W - self.patch_size,  box_mass[1])

                # avoid case where mass box size > patch size
                min_poss_h, max_poss_h = sorted([min_poss_h, max_poss_h])
                min_poss_w, max_poss_w = sorted([min_poss_w, max_poss_w])
                h2, w2 = random.randint(min_poss_h, max_poss_h), random.randint(min_poss_w, max_poss_w)
                box_patch = (h2, w2, h2 + self.patch_size, w2 + self.patch_size)
                labels.append(label)
            else:
                box_patch = self.__sample_patch__(cur_img, mask)
                labels.append(0)
            patch = cur_img[box_patch[0]:box_patch[2], box_patch[1]:box_patch[3]]
            if self.uniform_norm:
                patch = rescale_intensity(patch, out_range=(0, 1))
            img.append(Image.fromarray(patch))
            image_paths.append(img_path)

        if self.transform is not None:
            if self.contrastive:
                img = img[0]
            img_t = self.transform(img)
        labels = torch.LongTensor(labels)

        if self.temporal:
            # (T, H, W) -> (1, T, H, W)
            img_t = torch.unsqueeze(img_t, dim=0)

        if get_info:
            return img_t, labels, view, image_paths, dcm_path
        else:
            return img_t, labels


def get_dataset(meta, transform, args, test=False, get_info=False):
    # fix load from npz flag
    # args.load_from_npz = True
    num_slice = 'all' if test else args.num_slice
    if args.patch_lv:
        return BCS_DBT_patch_cls(meta, load_from_raw=args.load_from_raw,
                                use_otsu=args.use_otsu, transform=transform, binary=args.binary,
                                subset=args.subset, mid_slice=args.mid_slice, 
                                ignore_action=args.ignore_action,
                                num_slice=num_slice, sampling_gap=args.sampling_gap, fix_gap=args.fix_gap,
                                subset_ratio=args.subset_ratio, patch_size=args.patch_size, 
                                patch_eps=args.patch_eps, mass_eps=args.mass_eps, 
                                contrastive=args.contrastive, positive_range=args.positive_range,
                                load_from_npz=args.load_from_npz, temporal=args.temporal_model,
                                ignore_abnormal=args.ignore_abnormal, patch_cnt=args.patch_cnt,
                                get_info=get_info)
    else:
        return BCS_DBT_cls(meta, load_from_raw=args.load_from_raw,
                            use_otsu=args.use_otsu, transform=transform, binary=args.binary,
                            subset=args.subset, mid_slice=args.mid_slice, ignore_action=args.ignore_action,
                            num_slice=num_slice, sampling_gap=args.sampling_gap, fix_gap=args.fix_gap,
                            subset_ratio=args.subset_ratio, uniform_norm=args.uniform_norm,
                            pick_mass_slice=args.pick_mass_slice, contrastive=args.contrastive,
                            load_from_npz=args.load_from_npz, temporal=args.temporal_model,
                            positive_range=args.positive_range, inter_slice=args.inter_slice,
                            dino=args.dino, inter_view=args.inter_view, inter_patient=args.inter_patient,
                            inter_study=args.inter_study, ignore_abnormal=args.ignore_abnormal,
                            get_info=get_info)



if __name__ == '__main__':
    transform = transforms.Compose([
        ResizeLongSide((512, 512)),
        Padding((512, 512)),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((39.5281), (38.7750)),
    ])
    dataset = BCS_DBT_cls('data/csv/BCS-DBT_train_label.csv', load_from_raw=False,
                            use_otsu=True, transform=transform, binary=True,
                            subset=True, mid_slice=False)
    
    img, label = dataset.__getitem__(10)
    print(torch.mean(img), label)
