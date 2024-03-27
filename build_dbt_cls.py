import os
import json
import time
import argparse
from glob import glob
from dataset.duke_dbt_data import dcmread_image
import pandas as pd
import random
import numpy as np
import cv2
from multiprocessing import Pool
from functools import partial
from utils import convert_seconds

def split_dataset(df, seed=0):
    patient_set = set(df['PatientID'])

    normal_patient = []
    actionable_patient = []
    benign_patient = []
    cancer_patient = []
    for p in patient_set:
        # split data by patient
        p_df = df.loc[df['PatientID'] == p]
        if 1 in set(p_df['Cancer']):
            cancer_patient.append(p)
        elif 1 in set(p_df['Benign']):
            benign_patient.append(p)
        elif 1 in set(p_df['Actionable']):
            actionable_patient.append(p)
        else:
            normal_patient.append(p)

    print(len(normal_patient), len(actionable_patient), len(benign_patient), len(cancer_patient))
    random.seed = seed
    random.shuffle(normal_patient)
    random.shuffle(actionable_patient)
    random.shuffle(benign_patient)
    random.shuffle(cancer_patient)
    train_split_normal = int(len(normal_patient) * 0.7)
    val_split_normal = int(len(normal_patient) * 0.8)
    train_split_actionable = int(len(actionable_patient) * 0.7)
    val_split_actionable = int(len(actionable_patient) * 0.8)
    train_split_benign = int(len(benign_patient) * 0.7)
    val_split_benign = int(len(benign_patient) * 0.8)
    train_split_cancer = int(len(cancer_patient) * 0.7)
    val_split_cancer = int(len(cancer_patient) * 0.8)
    train_patient = normal_patient[:train_split_normal] + actionable_patient[:train_split_actionable] + benign_patient[:train_split_benign] + cancer_patient[:train_split_cancer]
    val_patient = normal_patient[train_split_normal:val_split_normal] + actionable_patient[train_split_actionable:val_split_actionable] + benign_patient[train_split_benign:val_split_benign] + cancer_patient[train_split_cancer:val_split_cancer]
    test_patient = normal_patient[val_split_normal:] + actionable_patient[val_split_actionable:] + benign_patient[val_split_benign:] + cancer_patient[val_split_cancer:]

    train_idx = []
    for p in train_patient:
        for idx in df.loc[df['PatientID'] == p].index:
            train_idx.append(idx)
    val_idx = []
    for p in val_patient:
        for idx in df.loc[df['PatientID'] == p].index:
            val_idx.append(idx)
    test_idx = []
    for p in test_patient:
        for idx in df.loc[df['PatientID'] == p].index:
            test_idx.append(idx)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    train_df.to_csv('./data/csv/BCS-DBT_train_label_v2.csv')
    val_df.to_csv('./data/csv/BCS-DBT_val_label_v2.csv')
    test_df.to_csv('./data/csv/BCS-DBT_test_label_v2.csv')


def otsu_cut(img, view):
    median = np.median(img)
    ret, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    B = np.argwhere(thresh)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
    return ystart, xstart, ystop, xstop

def save_dbt_to_png(df, target_size=512, args=None):
    cnt = []
    duration = []
    failed = []
    pid = os.getpid()
    df_box = pd.read_csv('data/csv/BCS-DBT boxes-train-v2.csv')
    box_resized = {}
    mean, std = [], []
    orig_size = {}
    for i in range(len(df)):
        st = time.time()
        entry = df.iloc[i]
        dcm_path = os.path.join("./data", entry["classic_path"])
        png_folder = dcm_path.replace('1-1.dcm', 'png')
        npz_dest = dcm_path.replace('1-1.dcm', '1-1.npz')
        otsu_npz_dest = dcm_path.replace('1-1.dcm', '1-1_otsu.npz')
        os.makedirs(png_folder, exist_ok=True)
        sid = entry['StudyUID']
        view = entry["View"]
        try:
            image = dcmread_image(fp=dcm_path, view=view)
            _, H, W = image.shape
            if os.path.exists(png_folder) and len(os.listdir(png_folder)) == 2*image.shape[0] and args.skip:
                png_list = sorted([p for p in os.listdir(png_folder) if 'otsu' not in p])
                probe_img = os.path.join(png_folder, png_list[-1])
                im = cv2.imread(probe_img, cv2.IMREAD_GRAYSCALE)
                im_h, im_w = im.shape
                cur_ratio = (im_h / im_w) - 1
                orig_ratio = (H / W) - 1
                short_side_idx = np.argmin((H, W))
                if im.shape[short_side_idx] == target_size and (cur_ratio * orig_ratio > 0):
                    with open(f'./tmp/proc_{pid}.out', 'a') as fp:
                        fp.write(f'Skip volume #{i}: {dcm_path}, continue...\n')
                    continue
            image = (image // 257).astype(np.uint8)
            cnt.append(image.shape[0])

            
            # resize short side to target_size
            if H < W:
                h = target_size
                ratio = h / H
                w = round(W * ratio)
            else:
                w = target_size
                ratio = w / W
                h = round(H * ratio)
            if entry['Benign'] or entry['Cancer']:
                box = df_box[df_box['StudyUID'] == sid]
                box = box[box['View'] == view]
                box_index = int(box.index[0])
                info = box[['X', 'Y', 'Width', 'Height']].to_dict('list')
                for k in info.keys():
                    info[k] = [round(v * ratio) for v in info[k]]
                box_resized[box_index] = info
            arr = []
            arr_otsu = []
            otsu_box = None
            for idx in range(image.shape[0]):
                im = image[idx]
                im = cv2.resize(im, (w, h))
                arr.append(im)
                dest = os.path.join(png_folder, f'slice_{str(idx).zfill(4)}.png')
                cv2.imwrite(dest, im)

                if otsu_box is None:
                    otsu_box = otsu_cut(im, view)
                otsu = im[otsu_box[0]:otsu_box[2], otsu_box[1]:otsu_box[3]]
                arr_otsu.append(otsu)
                otsu_dest = os.path.join(png_folder, f'slice_{str(idx).zfill(4)}_otsu.png')
                cv2.imwrite(otsu_dest, otsu)
            np.savez_compressed(npz_dest, arr=np.array(arr))
            np.savez_compressed(otsu_npz_dest, arr=np.array(arr_otsu))
            mean.append(np.mean(np.array(arr)))
            std.append(np.std(np.array(arr)))
        except Exception as e:
            with open(f'./tmp/proc_{pid}.out', 'a') as fp:
                fp.write(f'!!! Failed on volume {dcm_path} due to {e}. continue...\n')
            failed.append(i)
            continue
        et = time.time()
        duration.append(et - st)
        # etd = (np.mean(duration) * (len(df) - i)) / 60
        etd = convert_seconds(np.mean(duration) * (len(df) - i))
        log_str = f'PID: {pid}\tIDX: {i}\tAvg. #slices: {np.mean(cnt):.2f}\tAvg. time: {np.mean(duration):.2f}\tMean: {np.mean(mean):.4f}\tstd: {np.mean(std):.4f}\tETD: {etd}\n'
        with open(f'./tmp/proc_{pid}.out', 'a') as fp:
            fp.write(log_str)
    with open(f'./logs/resize_{target_size}_box_proc_{pid}.json', 'w') as fp:
        json.dump(box_resized, fp)
    with open(f'./logs/orig_size_{pid}.json', 'w') as fp:
        json.dump(orig_size, fp)
    return failed

parser = argparse.ArgumentParser(description='visualize features')
parser.add_argument('--size', type=int, default=768)
parser.add_argument('--nt', type=int, default=8)
parser.add_argument('--skip', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    df_paths = pd.read_csv('data/csv/BCS-DBT file-paths-train-v2.csv')
    df_label = pd.read_csv('data/csv/BCS-DBT labels-train-v2.csv')
    primary_key = ("PatientID", "StudyUID", "View")
    df_merge = pd.merge(df_paths, df_label, on=primary_key)
    target_size = args.size
    # split_dataset(df_merge)

    NT = args.nt
    df_list = [df_merge.iloc[list(range(p, len(df_merge), NT))] for p in range(NT)]

    with Pool(NT) as p:
        res = p.map(partial(save_dbt_to_png, target_size=target_size, args=args), df_list)
    with open('logs/failed_pre_process.json', 'w') as f:
        json.dump(res, f)

