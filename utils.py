import pandas as pd
import numpy as np

import torch
from nltk.corpus import stopwords
from sklearn import metrics

import re
import os
import json
import random


stops = set(stopwords.words("english"))
regex_punctuation = re.compile('[\',\.\-/\n]')
regex_alphanum = re.compile('[^a-zA-Z0-9 ]')
regex_num = re.compile('\d[\d ]+')
regex_spaces = re.compile('\s+')


def bin_age(age):
    if age < 25:
        return '18-25'
    elif age < 45:
        return '25-45'
    elif age < 65:
        return '45-65'
    elif age < 89:
        return '65-89'
    else:
        return '89+'


def clean_text(text):
    text = text.lower().strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    text = re.sub(regex_punctuation, ' ', text)
    text = re.sub(regex_alphanum, '', text)
    text = re.sub(regex_num, ' 0 ', text)
    text = re.sub(regex_spaces, ' ', text)
    return text.strip()

def text2words(text):
    words = text.split()
    words = [w for w in words if not w in stops]
    return words


def convert_icd_group(icd):
    icd = str(icd)
    if icd.startswith('V'):
        return 19
    if icd.startswith('E'):
        return 20
    icd = int(icd[:3])
    if icd <= 139:
        return 1
    elif icd <= 239:
        return 2
    elif icd <= 279:
        return 3
    elif icd <= 289:
        return 4
    elif icd <= 319:
        return 5
    elif icd <= 389:
        return 6
    elif icd <= 459:
        return 7
    elif icd <= 519:
        return 8
    elif icd <= 579:
        return 9
    elif icd < 629:
        return 10
    elif icd <= 679:
        return 11
    elif icd <= 709:
        return 12
    elif icd <= 739:
        return 13
    elif icd <= 759:
        return 14
    elif icd <= 779:
        return np.nan
    elif icd <= 789:
        return 15
    elif icd <= 796:
        return 16
    elif icd <= 799:
        return 17
    else:
        return 18


def cal_metric(y_true, probs):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, probs)
    optimal_idx = np.argmax(np.sqrt(tpr * (1-fpr)))
    optimal_threshold = thresholds[optimal_idx]
    preds = (probs > optimal_threshold).astype(int)
    auc = metrics.roc_auc_score(y_true, probs)
    auprc = metrics.average_precision_score(y_true, probs)
    f1 = metrics.f1_score(y_true, preds)
    return f1, auc, auprc


def save_model(all_dict, name='best_model.pth'):
    model_dir = all_dict['args'].model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, name)
    torch.save(all_dict, model_path)


def load_model(model_dict, name='best_model.pth'):
    model = model_dict['model']
    model_dir = model_dict['args'].model_dir
    model_path = os.path.join(model_dir, name)
    if os.path.exists(model_path):
        all_dict = torch.load(model_path)
        model.load_state_dict(all_dict['state_dict'])
        return model, all_dict['best_metric'], all_dict['epoch']
    else:
        return model, 0, 1
    

def get_ids(split_json):
    splits = list(range(10))
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def get_ids2(split_json, seed):
    splits = list(range(10))
    random.Random(seed).shuffle(splits)
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def balance_samples(df, times, task):
    df_pos = df[df[task] == 1]
    df_neg = df[df[task] == 0]
    df_neg = df_neg.sample(n=times * len(df_pos), random_state=42)
    df = pd.concat([df_pos, df_neg]).sort_values('hadm_id')
    return df


def mkdir(d):
    path = d.split('/')
    for i in range(len(path)):
        d = '/'.join(path[:i+1])
        if not os.path.exists(d):
            os.mkdir(d)


def csv_split(line, sc=','):
    res = []
    inside = 0
    s = ''
    for c in line:
        if inside == 0 and c == sc:
            res.append(s)
            s = ''
        else:
            if c == '"':
                inside = 1 - inside
            s = s + c
    res.append(s)
    return res
