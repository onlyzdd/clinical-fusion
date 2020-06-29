import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from gensim.models.doc2vec import Doc2Vec

import argparse
import json
import os
import time
import warnings

from utils import balance_samples, cal_metric, get_ids, get_ids2, text2words

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mortality')
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--inputs', type=int, default=4) # 1-7
    parser.add_argument('--seed', type=int, default=42) # random seed
    args = parser.parse_args()
    return args


def train_test_base(X_train, X_test, y_train, y_test, task, name, inputs, seed):
    mtl = 1 if y_test.shape[1] > 1 else 0 # multi-label
    if name == 'lr':
        # print('Start training Logistic Regression:')
        model = LogisticRegression()
    else:
        # print('Start training Random Forest:')
        model = RandomForestClassifier()
    if mtl:
        model = OneVsRestClassifier(model)
    else:
        y_train, y_test = y_train[:, 0], y_test[:, 0]
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    # print('Running time:', t1 - t0)
    probs = model.predict_proba(X_test)
    metrics = []
    if mtl:
        for idx in range(y_test.shape[1]):
            metric = cal_metric(y_test[:, idx], probs[:, idx])
            # print(idx + 1, metric)
            metrics.append(metric)
        # print('Avg', np.mean(metrics, axis=0).tolist())
    else:
        metric = cal_metric(y_test, probs[:, 1])
        f1, auc, aupr = metric
        # print(metric)
        print(f'{task},{name},{inputs},{seed},{f1},{auc},{aupr}')


if __name__ == '__main__':
    args = parse_args()
    task = args.task
    model = args.model
    inputs = args.inputs
    seed = args.seed
    # print('Running task %s using inputs %d...' % (task, inputs))
    train_ids, val_ids, test_ids = get_ids2('data/processed/files/splits.json', seed)

    df = pd.read_csv('data/processed/%s.csv' % task).sort_values('hadm_id')

    if task != 'labels_icd' and task != 'los_bin':
        pass
    train_ids = np.intersect1d(train_ids, df['hadm_id'].tolist())
    test_ids = np.intersect1d(test_ids, df['hadm_id'].tolist())

    choices = '{0:b}'.format(inputs).rjust(3, '0')
    X_train, X_test = [], []

    if choices[0] == '1':
        # print('Loading notes...')
        df_notes = pd.read_csv('data/processed/earlynotes.csv').sort_values('hadm_id')
        doc2vec = Doc2Vec.load('models/doc2vec.model')
        df_notes['text'] = df_notes['text'].astype(str)
        df_notes['vector'] = df_notes['text'].apply(lambda note: doc2vec.infer_vector(text2words(note)))
        df_notes = df_notes.groupby('hadm_id')['vector'].apply(list).reset_index()
        df_notes['vector'] = df_notes['vector'].apply(lambda notes: np.mean(notes, axis=0))
        df_notes_col = 'vector'

        X_train_notes = df_notes[df_notes['hadm_id'].isin(train_ids)][df_notes_col].to_list()
        X_test_notes = df_notes[df_notes['hadm_id'].isin(test_ids)][df_notes_col].to_list()

        X_train.append(X_train_notes)
        X_test.append(X_test_notes)
    if choices[1] == '1':
        # print('Loading temporal data...')
        df_temporal = pd.read_csv('data/processed/features.csv').drop('charttime', axis=1)
        temporal_mm_dict = json.load(open('data/processed/files/feature_mm_dict.json'))
        for col in df_temporal.columns[1:]:
            col_min, col_max = temporal_mm_dict[col]
            df_temporal[col] = (df_temporal[col] - col_min) / (col_max - col_min)
        df_temporal = df_temporal.groupby(
            'hadm_id').agg(['mean', 'count', 'max', 'min', 'std'])
        df_temporal.columns = ['_'.join(col).strip()
                                for col in df_temporal.columns.values]
        df_temporal.fillna(0, inplace=True)
        df_temporal = df_temporal.reset_index().sort_values('hadm_id')
        df_temporal_cols = df_temporal.columns[1:]
        X_train_temporal = df_temporal[df_temporal['hadm_id'].isin(train_ids)][df_temporal_cols].to_numpy()
        X_test_temporal = df_temporal[df_temporal['hadm_id'].isin(test_ids)][df_temporal_cols].to_numpy()
        X_train.append(X_train_temporal)
        X_test.append(X_test_temporal)
    if choices[2] == '1':
        # print('Loading demographics...')
        df_demo = pd.read_csv('data/processed/demo.csv').sort_values('hadm_id')
        df_demo = pd.get_dummies(df_demo, drop_first=True)
        df_demo_cols = df_demo.columns[1:]
        X_train_demo = df_demo[df_demo['hadm_id'].isin(train_ids)][df_demo_cols].to_numpy()
        X_test_demo = df_demo[df_demo['hadm_id'].isin(test_ids)][df_demo_cols].to_numpy()
        X_train.append(X_train_demo)
        X_test.append(X_test_demo)

    # print('Done.')
    df_cols = df.columns[1:]
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    y_train = df[df['hadm_id'].isin(train_ids)][df_cols].to_numpy()
    y_test = df[df['hadm_id'].isin(test_ids)][df_cols].to_numpy()

    if model == 'all':
        train_test_base(X_train, X_test, y_train, y_test, task, 'lr', inputs, seed)
        train_test_base(X_train, X_test, y_train, y_test, task, 'rf', inputs, seed)
    else:
        train_test_base(X_train, X_test, y_train, y_test, task, model, inputs, seed)