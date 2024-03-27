import os

import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import numpy as np
from utils import get_specificity_with_sensitivity, plot_auc_roc_img
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve

def eval_patient(test_output, score=0.5, ab_slice_threshold=22, eval_patient_cnt=True, avg_logits=True, verbose=False, calibrate=False, get_fpr_tpr=False):
    patient_label = [] 
    patient_score = []
    patient_pred = []
    abnormal_cnt = []
    abnormal_recall = 0
    correct = 0
    for k, v in test_output.items():
        cur_label = 0
        cur_pred = 0
        cur_cnt = 0
        logits = []
        for (output, target) in v:
            logits.append(output)
            if target == 1:
                cur_label = 1
            if output[1] > score:
                cur_cnt += 1
        logits = np.stack(logits, axis=0)
        if eval_patient_cnt:
            if cur_cnt >= ab_slice_threshold:
                cur_pred = 1
            if cur_pred == 1:
                patient_score.append(logits[:, 1].max())
            else:
                patient_score.append(logits[:, 1].min())
        else:
            if avg_logits:
                patient_logits = np.mean(logits, axis=0)
            else:
                patient_logits_idx = np.argmax(logits[:, 1])
                patient_logits = logits[patient_logits_idx]
            # print(patient_logits)
            if calibrate:
                # re-calibrate the logits based on score threshold
                gap = patient_logits[1] - score
                calibrated_patient_logits = np.clip(0.5+gap, a_min=0.0, a_max=1.0)
                patient_logits = np.array([1-calibrated_patient_logits, calibrated_patient_logits])
            if patient_logits[1] > score:
                cur_pred = 1
            patient_score.append(patient_logits[1])
        abnormal_cnt.append(cur_cnt)
        abnormal_recall += 1 if cur_label == cur_pred == 1 else 0
        correct += 1 if cur_label == cur_pred else 0
        patient_label.append(cur_label)
        patient_pred.append(cur_pred)

    acc = np.sum(np.array(patient_label) == np.array(patient_pred)) / len(patient_pred)
    auc = 100 * roc_auc_score(patient_label, patient_score)
    sen_87 = 100 * get_specificity_with_sensitivity(patient_label, patient_score, 0.87)
    sen_80 = 100 * get_specificity_with_sensitivity(patient_label, patient_score, 0.80)
    abnormal_recall = abnormal_recall / sum(patient_label)
    metric_dict = classification_report(patient_label, patient_pred, output_dict=True, zero_division=0.0)
    if verbose:
        print(classification_report(patient_label, patient_pred, digits=5, zero_division=0.0))
        print(f'AUC: {auc:.2f}')
        print(f'Sensitivity at 87%: {sen_87:.2f}')
        print(f'Sensitivity at 80%: {sen_80:.2f}')
        plot = plot_auc_roc_img(patient_label, patient_score, auc)
        plt.show()
    if get_fpr_tpr:
        fpr, tpr, thresholds = roc_curve(patient_label, patient_score)
        return acc, auc, sen_87, sen_80, metric_dict, fpr, tpr
    return acc, auc, sen_87, sen_80, metric_dict

def find_optimal_score(test_output, verbose=False):
    accs = []
    aucs = []
    sen_87s = []
    sen_80s = []
    recalls = []
    normal_recalls = []
    macro_precisions = []
    macro_recalls = []

    scores = np.arange(0, 1.01, 0.01)
    for score in scores:
        acc, auc, sen_87, sen_80, metric_dict = eval_patient(test_output, score=score, avg_logits=False, eval_patient_cnt=False, verbose=False, calibrate=False)
        accs.append(acc)
        aucs.append(auc)
        sen_87s.append(sen_87)
        sen_80s.append(sen_80)
        recalls.append(metric_dict['1']['recall'])
        normal_recalls.append(metric_dict['0']['recall'])
        macro_precisions.append(metric_dict['macro avg']['precision'])
        macro_recalls.append(metric_dict['macro avg']['recall'])
    max_idx = np.argmax(macro_recalls)
    min_dix = np.argmin(np.abs(np.array(macro_recalls) - np.array(recalls)))
    if verbose:
        print(np.max(aucs))
        plt.figure(figsize=(12, 12))
        plt.plot(scores, accs, label='Accuracy')
        plt.plot(scores, aucs, label='AUC')
        plt.plot(scores, sen_87s, label='Sensitivity at 87%')
        plt.plot(scores, sen_80s, label='Sensitivity at 80%')
        plt.plot(scores, normal_recalls, label='Normal Recall')
        plt.plot(scores, recalls, label='Abnormal Recall')
        plt.plot(scores, macro_precisions, label='Macro Precision')
        plt.plot(scores, macro_recalls, label='Macro Recall')
        print(scores[max_idx], macro_recalls[max_idx], aucs[max_idx], sen_87s[max_idx], sen_80s[max_idx])
        print(scores[min_dix], macro_recalls[min_dix], aucs[min_dix], sen_87s[min_dix], sen_80s[min_dix])
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
    return scores[min_dix]

def get_slice_result(test_output):
    slice_label = []
    slice_pred = []
    for k, v in test_output.items():
        for (output, target) in v:
            slice_label.append(target)
            slice_pred.append(output[1])
    return slice_label, slice_pred


def eval_patient(feat_path):
    test_output = np.load(feat_path, allow_pickle=True)['vol2pred'][()]
    score = find_optimal_score(test_output)
    print(score)
    acc, auc, sen_87, sen_80, metric_dict, fpr, tpr = eval_patient(test_output, score=score, avg_logits=False, calibrate=False, eval_patient_cnt=False, verbose=True, get_fpr_tpr=True)
    return acc, auc, sen_87, sen_80, metric_dict, fpr, tpr

if __name__ == '__main__':
    feat_path = '<Replace with extracted feature>'
    assert os.path.exists(feat_path)
    eval_patient(feat_path)
