#!/usr/bin/env python
# coding=utf-8
import pandas as pd

import sys

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm

sys.path.append('./tools')
import parse, py_op

args = parse.args
py_op.mkdir(args.data_dir)
py_op.mkdir(args.result_dir)
py_op.mkdir(args.lab_test_data_dir)
py_op.mkdir(args.lab_test_initial_dir)
py_op.mkdir(args.lab_test_resample_dir)
py_op.mkdir(args.lab_test_file_dir)
py_op.mkdir(args.lab_test_result_dir)
data_csv = os.path.join(args.lab_test_data_dir, 'features.csv')
label_csv = os.path.join(args.lab_test_data_dir, 'mortality.csv')
demo_csv = os.path.join(args.lab_test_data_dir, 'demo.csv')
note_csv = os.path.join(args.lab_test_data_dir, 'earlynotes.csv')
selected_indices = []

def time_to_second(t):
    t = str(t).replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    return int(t)

def get_time(t):
    try:
        t = float(t)
        return t
    except:
        t = str(t).replace('"', '')
        t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
        t = int(t/3600)
        return t

def generate_file_for_each_patient():
    '''
    create a lab test file for each patient
    '''
    global selected_indices
    initial_dir = args.lab_test_initial_dir
    os.system('rm -r ' + initial_dir)
    py_op.mkdir(initial_dir)
    label_dict = py_op.myreadjson(os.path.join(args.lab_test_file_dir, 'label_dict.json'))
    for i_line, line in enumerate(open(data_csv)):
        if i_line % 10000 == 0:
            print( i_line)
        if i_line:
            line_data = line.strip().split(',')
            if line_data[0] not in label_dict:
                continue

            assert len(line_data) == len(feat_list)
            new_line_data = [line_data[i_feat] for i_feat in selected_indices]
            new_line = ','.join(new_line_data)

            p_file = os.path.join(initial_dir, line_data[0] + '.csv')
            if not os.path.exists(p_file):
                wf = open(p_file, 'w') 
                wf.write(new_head)
                wf.close()
            wf = open(p_file, 'a')
            wf.write('\n' + new_line)
            wf.close()
        else:
            feat_list = py_op.csv_split(line.strip())
            feat_list = [f.strip('"') for f in feat_list]
            print('There are {:d} features.'.format(len(feat_list)))
            print(feat_list)
            if len(selected_indices) == 0:
                selected_indices = range(1, len(feat_list))
                selected_feat_list = [feat_list[i_feat].replace('"','').replace(',', ';') for i_feat in selected_indices]
                new_head = ','.join(selected_feat_list)


def resample_lab_test_data(delta=1, ignore_time=-48):
    '''
    resample data so that the records have same time intervals
    '''
    resample_dir = args.lab_test_resample_dir
    initial_dir = args.lab_test_initial_dir

    os.system('rm -r ' + resample_dir)
    py_op.mkdir(resample_dir)

    count_intervals = [0, 0]
    count_dict = dict()
    two_sets = [set(), set()]
    for i_fi, fi in enumerate(tqdm(os.listdir(initial_dir))):
        time_line_dict = dict()
        for i_line, line in enumerate(open(os.path.join(initial_dir, fi))):
            if i_line:
                if len(line.strip()) == 0:
                    continue
                line_data = line.strip().split(',')
                assert len(line_data) == len(feat_list)
                ctime = get_time(line_data[0])
                ctime = delta * int(float(ctime) / delta)
                if ctime not in time_line_dict:
                    time_line_dict[ctime] = []
                time_line_dict[ctime].append(line_data)
            else:
                feat_list = line.strip().split(',')
                feat_list[0] = 'time'

        if len(time_line_dict) < 10:
            # continue
            pass

        wf = open(os.path.join(resample_dir, fi), 'w')
        wf.write(','.join(feat_list))
        last_time = None
        vis = 0
        max_t = max(time_line_dict)
        for t in sorted(time_line_dict):
            if t - max_t < ignore_time:
                continue
            line_list = time_line_dict[t]
            new_line = line_list[0]
            for line_data in line_list:
                for iv, v in enumerate(line_data):
                    if len(v.strip()):
                        new_line[iv] = v
            new_line[0] = str(t - max_t)
            new_line = '\n' + ','.join(new_line) 
            wf.write(new_line)

            if last_time is not None:
                delta_t = t - last_time
                if delta_t > delta:
                    vis = 1
                    # break
                    count_intervals[0] += 1
                    count_dict[t - last_time] = count_dict.get(t - last_time, 0) + 1
                    two_sets[0].add(fi)
                two_sets[1].add(fi)
                count_intervals[1] += 1
            last_time = t
        wf.close()
        if vis:
            # os.system('rm ' + os.path.join(resample_dir, fi))
            pass
    print('There are {:d}/{:d} collections data with intervals > {:d}.'.format(count_intervals[0], count_intervals[1], delta))
    print('There are {:d}/{:d} patients with intervals > {:d}.'.format(len(two_sets[0]), len(two_sets[1]), delta))
    # print(count_dict)

def generate_feature_mm_dict():
    resample_dir = args.lab_test_resample_dir
    files = sorted(glob(os.path.join(resample_dir, '*')))
    feature_value_dict = dict()
    feature_missing_dict = dict()
    for ifi, fi in enumerate(tqdm(files)):
        if 'csv' not in fi:
            continue
        for iline, line in enumerate(open(fi)):
            line = line.strip()
            if iline == 0:
                feat_list = line.split(',')
            else:
                data = line.split(',')
                for iv, v in enumerate(data):
                    if v in ['NA', '']:
                        continue
                    else:
                        feat = feat_list[iv]
                        if feat not in feature_value_dict:
                            feature_value_dict[feat] = []
                        feature_value_dict[feat].append(float(v))
    feature_mm_dict = dict()
    feature_ms_dict = dict()

    feature_range_dict = dict()
    len_time = max([len(v) for v in feature_value_dict.values()])
    for feat, vs in feature_value_dict.items():
        vs = sorted(vs)
        value_split = []
        for i in range(args.split_num):
            n = int(i * len(vs) / args.split_num)
            value_split.append(vs[n])
        value_split.append(vs[-1])
        feature_range_dict[feat] = value_split


        n = int(len(vs) / args.split_num)
        feature_mm_dict[feat] = [vs[n], vs[-n - 1]]
        feature_ms_dict[feat] = [np.mean(vs), np.std(vs)]

        feature_missing_dict[feat] = 1.0 - 1.0 * len(vs) / len_time
    
    print('Recommed features: ')
    for feat, mr in feature_missing_dict.items():
        if mr < 0.9:
            print(feat, mr)

    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'feature_mm_dict.json'), feature_mm_dict)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'feature_ms_dict.json'), feature_ms_dict)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'feature_list.json'), feat_list)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'feature_missing_dict.json'), feature_missing_dict)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'feature_value_dict_{:d}.json'.format(args.split_num)), feature_range_dict)

def split_data_to_ten_set():
    resample_dir = args.lab_test_resample_dir
    files = sorted(glob(os.path.join(resample_dir, '*')))
    np.random.shuffle(files)
    splits = []
    for i in range(10):
        st = int(len(files) * i / 10)
        en = int(len(files) * (i+1) / 10)
        splits.append(files[st:en])
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'splits.json'), splits)

def generate_label_dict():
    label_dict = dict()
    for i_line, line in enumerate(open(label_csv)):
        if i_line:
            data = line.strip().split(',')
            pid = data[0]
            label = ''.join(data[1:])
            pid = str(int(float(pid)))
            label_dict[pid] = label
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'label_dict.json'), label_dict)

def generate_demo_dict():
    demo_dict = dict()
    demo_index_dict = dict()
    for i_line, line in enumerate(open(demo_csv)):
        if i_line:
            data = line.strip().split(',')
            pid = str(int(float(data[0])))
            demo_dict[pid] = []
            for demo in data[1:]:
                if demo not in demo_index_dict:
                    demo_index_dict[demo] = len(demo_index_dict)
                demo_dict[pid].append(demo_index_dict[demo])
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'demo_dict.json'), demo_dict)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'demo_index_dict.json'), demo_index_dict)

def generate_unstructure_dict():
    unstructure_dict = dict()
    vocab_dict = dict()
    for i_line, line in enumerate(open(note_csv)):
        if i_line:
            data = line.strip().split(',')
            pid = data[0]
            content = ','.join(data[2:])
            for c in '*,"\':()?/#.':
                content = content.replace(c, '')
            content = content.strip().split()
            unstructure_dict[pid] = content
            for w in content:
                vocab_dict[w] = vocab_dict.get(w, 0) + 1
        if i_line % 1000 == 0:
            print(i_line)
    vocab_list = [''] + [v for v,c in vocab_dict.items() if c > 50]
    print('There are {:d}/{:d} vocabs.'.format(len(vocab_list), len(vocab_dict)))
    vocab_idx = { v:i for i,v in enumerate(vocab_list) }
    for pid, content in unstructure_dict.items():
        idx = []
        for v in content:
            if v in vocab_idx:
                idx.append(vocab_idx[v])
        unstructure_dict[pid] = idx
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'unstructure_dict.json'), unstructure_dict)
    py_op.mywritejson(os.path.join(args.lab_test_file_dir, 'vocab_list.json'), vocab_list)


def generate_note_dict():
    note_dict = dict()
    df_notes = pd.read_csv(note_csv)
    df_notes['hadm_id'] = df_notes['hadm_id'].astype(int)
    s_notes = df_notes.groupby('hadm_id')['text'].apply(list)
    with open('./data/processed/files/notes_dict.json', 'w') as f:
        f.write(s_notes.to_json())


def main():
    # generate_note_dict()
    generate_label_dict()
    # generate_demo_dict()

    # generate_file_for_each_patient()
    # resample_lab_test_data()
    # generate_feature_mm_dict()
    # split_data_to_ten_set()



if __name__ == '__main__':
    main()

