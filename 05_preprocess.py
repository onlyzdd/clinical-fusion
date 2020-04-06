import numpy as np
from tqdm import tqdm

import os
import time
import json
import argparse
from glob import glob

import utils


def parse_args():
    parser = argparse.ArgumentParser(description='preprocessing help')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='data dir')
    return parser.parse_args()


def get_time(t):
    try:
        t = float(t)
        return t
    except:
        t = str(t).replace('"', '')
        t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
        t = int(t/3600)
        return t

def generate_file_for_each_patient(args, features_csv):
    selected_indices = []
    initial_dir = args.initial_dir
    os.system('rm -r ' + initial_dir)
    utils.mkdir(initial_dir)
    for i_line, line in enumerate(open(features_csv)):
        if i_line % 10000 == 0:
            print( i_line)
        if i_line:
            line_data = line.strip().split(',')

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
            feat_list = utils.csv_split(line.strip())
            feat_list = [f.strip('"') for f in feat_list]
            print('There are {:d} features.'.format(len(feat_list)))
            print(feat_list)
            if len(selected_indices) == 0:
                selected_indices = range(1, len(feat_list))
                selected_feat_list = [feat_list[i_feat].replace('"','').replace(',', ';') for i_feat in selected_indices]
                new_head = ','.join(selected_feat_list)


def resample_data(args, delta=1, ignore_time=-48):
    resample_dir = args.resample_dir
    initial_dir = args.initial_dir

    os.system('rm -r ' + resample_dir)
    utils.mkdir(resample_dir)

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
                    count_intervals[0] += 1
                    count_dict[t - last_time] = count_dict.get(t - last_time, 0) + 1
                    two_sets[0].add(fi)
                two_sets[1].add(fi)
                count_intervals[1] += 1
            last_time = t
        wf.close()
    print('There are {:d}/{:d} collections data with intervals > {:d}.'.format(count_intervals[0], count_intervals[1], delta))
    print('There are {:d}/{:d} patients with intervals > {:d}.'.format(len(two_sets[0]), len(two_sets[1]), delta))


def generate_feature_dict(args):
    resample_dir = args.resample_dir
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
    
    json.dump(feature_mm_dict, open(os.path.join(args.files_dir, 'feature_mm_dict.json'), 'w'))
    json.dump(feature_ms_dict, open(os.path.join(args.files_dir, 'feature_ms_dict.json'), 'w'))
    json.dump(feat_list, open(os.path.join(args.files_dir, 'feature_list.json'), 'w'))
    json.dump(feature_missing_dict, open(os.path.join(args.files_dir, 'feature_missing_dict.json'), 'w'))
    json.dump(feature_range_dict, open(os.path.join(args.files_dir, 'feature_value_dict_{:d}.json'.format(args.split_num)), 'w'))


def split_data_to_ten_set(args):
    resample_dir = args.resample_dir
    files = sorted(glob(os.path.join(resample_dir, '*')))
    np.random.shuffle(files)
    splits = []
    for i in range(10):
        st = int(len(files) * i / 10)
        en = int(len(files) * (i+1) / 10)
        splits.append(files[st:en])
    json.dump(splits, open(os.path.join(args.files_dir, 'splits.json'), 'w'))


def generate_label_dict(args, task):
    label_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.data_dir, '%s.csv' % task))):
        if i_line:
            data = line.strip().split(',')
            pid = data[0]
            label = ''.join(data[1:])
            pid = str(int(float(pid)))
            label_dict[pid] = label
    json.dump(label_dict, open(os.path.join(args.files_dir, '%s_dict.json' % task), 'w'))


def generate_demo_dict(args, demo_csv):
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
    json.dump(demo_dict, open(os.path.join(args.files_dir, 'demo_dict.json'), 'w'))
    json.dump(demo_index_dict, open(os.path.join(args.files_dir, 'demo_index_dict.json'), 'w'))


def main():
    args = parse_args()
    args.files_dir = os.path.join(args.data_dir, 'files')
    args.initial_dir = os.path.join(args.data_dir, 'initial_data')
    args.resample_dir = os.path.join(args.data_dir, 'resample_dir')
    args.split_num = 4000
    utils.mkdir(args.files_dir)
    utils.mkdir(args.initial_dir)
    utils.mkdir(args.resample_dir)
    features_csv = os.path.join(args.data_dir, 'features.csv')
    demo_csv = os.path.join(args.data_dir, 'demo.csv')
    for task in ['mortality', 'readmit', 'llos']:
        generate_label_dict(args, task)
    generate_demo_dict(args, demo_csv)
    generate_file_for_each_patient(args, features_csv)
    resample_data(args)
    generate_feature_dict(args)
    split_data_to_ten_set(args)


if __name__ == '__main__':
    main()
