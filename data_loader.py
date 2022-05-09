import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

vector_dict = json.load(open('data/processed/files/vector_dict.json', 'r'))


def find_index(v, vs, i=0, j=-1):
    if j == -1:
        j = len(vs) - 1

    if v > vs[j]:
        return j + 1
    elif v < vs[i]:
        return i
    elif j - i == 1:
        return j

    k = int((i + j)/2)
    if v <= vs[k]:
        return find_index(v, vs, i, k)
    else:
        return find_index(v, vs, k, j)


class DataBowl(Dataset):
    def __init__(self, args, files, phase='train'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.args = args
        self.phase = phase
        self.files = files
        self.feature_mm_dict = json.load(
            open(os.path.join(args.files_dir, 'feature_mm_dict.json'), 'r'))
        self.feature_value_dict = json.load(open(os.path.join(
            args.files_dir, 'feature_value_dict_%d.json' % args.split_num), 'r'))
        self.demo_dict = json.load(
            open(os.path.join(args.files_dir, 'demo_dict.json'), 'r'))
        self.label_dict = json.load(
            open(os.path.join(args.files_dir, '%s_dict.json' % args.task), 'r'))

        print('Use the last %d collections data' % args.n_visit)

    def map_input(self, value, feat_list, feat_index):
        index_start = (feat_index + 1) * (1 + self.args.split_num) + 1

        if value in ['NA', '']:
            return 0
        else:
            value = float(value)
            vs = self.feature_value_dict[feat_list[feat_index]][1:-1]
            v = find_index(value, vs) + index_start
            return v

    def map_output(self, value, feat_list, feat_index):
        if value in ['NA', '']:
            return 0
        else:
            value = float(value)
            minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
            if maxv <= minv:
                print(feat_list[feat_index], minv, maxv)
            assert maxv > minv
            v = (value - minv) / (maxv - minv)
            v = max(0, min(v, 1))
            return v

    def get_mm_item(self, idx):
        input_file = self.files[idx]
        pid = input_file.split('/')[-1].split('.')[0]

        with open(input_file) as f:
            input_data = f.read().strip().split('\n')

        time_list, input_list = [], []

        for iline in range(len(input_data)):
            inp = input_data[iline].strip()
            if iline == 0:
                feat_list = inp.split(',')
            else:
                in_vs = inp.split(',')
                ctime = int(inp.split(',')[0])
                input = []
                for i, iv in enumerate(in_vs):
                    if self.args.use_ve:
                        input.append(self.map_input(iv, feat_list, i))
                    else:
                        input.append(self.map_output(iv, feat_list, i))
                input_list.append(input)
                time_list.append(- int(ctime))

        if len(input_list) < self.args.n_visit:
            for _ in range(self.args.n_visit - len(input_list)):
                # pad empty visit
                vs = [0 for _ in range(self.args.input_size + 1)]
                input_list = [vs] + input_list
                time_list = [time_list[0]] + time_list
        else:
            if self.use_first_records:
                input_list = input_list[: self.args.n_visit]
                time_list = time_list[: self.args.n_visit]
            else:
                input_list = input_list[-self.args.n_visit:]
                time_list = time_list[-self.args.n_visit:]

        if self.args.value_embedding == 'no' or self.args.use_ve == 0:
            input_list = np.array(input_list, dtype=np.float32)
        else:
            input_list = np.array(input_list, dtype=np.int64)
        time_list = np.array(time_list, dtype=np.int64) + 1
        assert time_list.min() >= 0
        if self.args.value_embedding != 'no':
            input_list = input_list[:, 1:]
        else:
            input_list = input_list.transpose()

        label = np.array([int(l)
                          for l in self.label_dict[pid]], dtype=np.float32)
        # demo = np.array([self.demo_dict[pid] for _ in range(self.args.n_visit)], dtype=np.int64)
        demo = np.array(self.demo_dict.get(pid, 0), dtype=np.int64)

        # content = self.unstructure_dict.get(pid, [])
        # while len(content) < self.max_length:
        #     content.append(0)
        # content = content[: self.max_length]
        # content = np.array(content, dtype=np.int64)
        content = vector_dict[pid]
        while len(content) < 12:
            content.append([0] * 200)
        content = content[:12]
        content = np.array(content, dtype=np.float32)
        # content = np.mean(content, axis=0)

        return torch.from_numpy(input_list), torch.from_numpy(time_list), torch.from_numpy(demo), torch.from_numpy(content), torch.from_numpy(label), input_file

    def __getitem__(self, idx):
        return self.get_mm_item(idx)

    def __len__(self):
        return len(self.files)
