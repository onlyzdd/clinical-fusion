#!/usr/bin/env python
# coding=utf-8
import numpy as np

import os
import torch
from sklearn import metrics


def compute_nRMSE(pred, label, mask):
    '''
    same as 3dmice
    '''
    assert pred.shape == label.shape == mask.shape

    missing_indices = mask==1
    missing_pred = pred[missing_indices]
    missing_label = label[missing_indices]
    missing_rmse = np.sqrt(((missing_pred - missing_label) ** 2).mean())

    init_indices = mask==0
    init_pred = pred[init_indices]
    init_label = label[init_indices]
    init_rmse = np.sqrt(((init_pred - init_label) ** 2).mean())

    metric_list = [missing_rmse, init_rmse]
    for i in range(pred.shape[2]):
        apred = pred[:,:,i]
        alabel = label[:,:, i]
        amask = mask[:,:, i]

        mrmse, irmse = [], []
        for ip in range(len(apred)):
            ipred = apred[ip]
            ilabel = alabel[ip]
            imask = amask[ip]

            x = ilabel[imask>=0]
            if len(x) == 0:
                continue

            minv = ilabel[imask>=0].min()
            maxv = ilabel[imask>=0].max()
            if maxv == minv:
                continue

            init_indices = imask==0
            init_pred = ipred[init_indices]
            init_label = ilabel[init_indices]

            missing_indices = imask==1
            missing_pred = ipred[missing_indices]
            missing_label = ilabel[missing_indices]

            assert len(init_label) + len(missing_label) >= 2

            if len(init_pred) > 0:
                init_rmse = np.sqrt((((init_pred - init_label) / (maxv - minv)) ** 2).mean())
                irmse.append(init_rmse)

            if len(missing_pred) > 0:
                missing_rmse = np.sqrt((((missing_pred - missing_label)/ (maxv - minv)) ** 2).mean())
                mrmse.append(missing_rmse)

        metric_list.append(np.mean(mrmse))
        metric_list.append(np.mean(irmse))

    metric_list = np.array(metric_list)


    metric_list[0] = np.mean(metric_list[2:][::2])
    metric_list[1] = np.mean(metric_list[3:][::2])

    return metric_list


def save_model(p_dict):
    args = p_dict['args']
    model = p_dict['model']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'args': p_dict['args'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict 
            }
    torch.save(all_dict, args.model_path)

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    # p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    # for k,v in all_dict['state_dict'].items():
    #     p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])
    p_dict['model'].load_state_dict(all_dict['state_dict'])

def compute_auc(labels, probs):
    fpr, tpr, thr = metrics.roc_curve(labels, probs)
    return metrics.auc(fpr, tpr)

def compute_metric(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, probs)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    preds = [1 if prob >= optimal_threshold else 0 for prob in probs]
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
    precision = 1.0 * (tp / (tp + fp))
    sen = 1.0 * (tp / (tp + fn))  # recall
    spec = 1.0 * (tn / (tn + fp))
    f1 = metrics.f1_score(labels, preds)
    return precision, sen, spec, f1, auc, aupr
