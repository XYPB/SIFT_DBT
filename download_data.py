from multiprocessing import Pool
import os
import zipfile
import json
import time

from glob import glob
import shutil
from tciaclient.core import TCIAClient


def download_img(series_idx):
    data_dest_dir = './data/Breast-Cancer-Screening-DBT'
    collect_name = 'Breast-Cancer-Screening-DBT'
    tc = TCIAClient()
    series = tc.get_series(collection=collect_name)
    failed_list = []

    for idx in series_idx:
        s = series[idx]
        fname = str(idx).zfill(6) + f'_{collect_name}.zip'
        dest = os.path.join(data_dest_dir, fname)
        # if os.path.exists(dest):
        #     continue
        try:
            print(f'downloading {idx}...')
            download_res = tc.get_image(seriesInstanceUid = s["SeriesInstanceUID"],
                                        downloadPath = data_dest_dir, zipFileName = fname)
        except Exception as e:
            print(f'!!! Download failed due to error {e}')
            print(f'!!! {idx} download failed! removed & continue...\n')
            if os.path.exists(dest):
                os.remove(dest)
            failed_list.append(idx)
            # re-try after 1 sec
            time.sleep(0.1)
            continue
        if not download_res or not os.path.exists(dest):
            print(f'!!! {idx} download returned but failed! continue...\n')
            
            failed_list.append(idx)
            # re-try after 1 sec
            time.sleep(0.1)
            continue
    return failed_list


def unzip_imgs(series_idx):
    data_dest_dir = './data/Breast-Cancer-Screening-DBT'
    collect_name = 'Breast-Cancer-Screening-DBT'
    tc = TCIAClient()
    series = tc.get_series(collection=collect_name)
    unzip_failed_list = []

    for idx in series_idx:
        fname = str(idx).zfill(6) + f'_{collect_name}.zip'
        zip_file = os.path.join(data_dest_dir, fname)
        patientID = series[idx]['PatientID']
        studyInstanceUID = series[idx]['StudyInstanceUID']
        seriesInstanceUID = series[idx]['SeriesInstanceUID']
        unzip_dest = os.path.join(data_dest_dir, patientID, studyInstanceUID, seriesInstanceUID)
        unzip_file_dest = os.path.join(unzip_dest, '1-1.dcm')
        if os.path.exists(zip_file) and not os.path.exists(unzip_file_dest):
            print(f'unzip {zip_file}...')
            os.makedirs(unzip_dest, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dest)
                # rename to 1-1.dcm
                unzip_file = glob(os.path.join(unzip_dest, '*.dcm'))
                assert len(unzip_file) == 1
                unzip_file = unzip_file[0]
                shutil.move(unzip_file, unzip_file_dest)
            except Exception as e:
                print(f'!!! Error {e}')
                print(f'!!! Failed to unzip {zip_file}, removed, continue...')
                unzip_failed_list.append(int(zip_file.split('/')[-1].split('_')[0]))
                os.remove(zip_file)
                continue
    return unzip_failed_list


if __name__ == '__main__':
    recover_from = 0
    data_dest_dir = './data/Breast-Cancer-Screening-DBT'
    assert os.path.exists(data_dest_dir)
    collect_name = 'Breast-Cancer-Screening-DBT'
    
    tc = TCIAClient()
    series = tc.get_series(collection=collect_name)

    NT = 8
    num_imgs = len(series)
    sub_series_idx = [[j for j in range(i, num_imgs, NT)] for i in range(recover_from, recover_from+NT, 1)]
    # sub_series_idx = []
    with Pool(NT) as p:
        results = p.map(download_img, sub_series_idx)
        p.close()
        p.join()
        with open(f'./logs/missing_idx.json', 'w') as fp:
            json.dump(results, fp)

    sub_series_idx = [[j for j in range(i, num_imgs, NT)] for i in range(NT)]
    # sub_series_idx = []
    with Pool(NT) as p:
        results = p.map(unzip_imgs, sub_series_idx)
        p.close()
        p.join()
        with open(f'./logs/unzip_failed_idx.json', 'w') as fp:
            json.dump(results, fp)
