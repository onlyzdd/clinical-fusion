import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from gensim.models.doc2vec import Doc2Vec

import os
import json

from utils import text2words


class FusionDataset(Dataset):
    '''Fusion dataset'''
    def __init__(self, adm_ids, args):
        super(FusionDataset, self).__init__()
        print('Start loading...')
        self.doc2vec = Doc2Vec.load(args.doc2vec_path)
        self.adm_ids = adm_ids
        self.root_dir = args.root_dir
        self.n_notes = 6
        self.n_temporal = 24
        self.n_features = 26
        self.files_dir = os.path.join(args.root_dir, 'files')
        self.data_dir = os.path.join(args.root_dir, 'resample_data')
        self.label_dict = json.load(open(os.path.join(self.files_dir, 'label_dict.json')))
        self.demo_dict = json.load(open(os.path.join(self.files_dir, 'demo_dict.json')))
        self.feature_mm_dict = json.load(open(os.path.join(self.files_dir, 'feature_mm_dict.json')))
        self.notes_dict = json.load(open(os.path.join(self.files_dir, 'notes_dict.json')))
        print('End loading...')

    def scaler(self, v, feature_list, idx):
        if v == '':
            return 0
        else:
            v = float(v)
            minv, maxv = self.feature_mm_dict[feature_list[idx]]
            v = (v - minv) / (maxv - minv + 1e-10)
            return v

    def get_temporal(self, adm_id):
        input_file = os.path.join(self.data_dir, '%s.csv' % adm_id)
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')
        input_list = []
        for iline, line in enumerate(input_data):
            if iline == 0:
                feature_list = line.strip().split(',')
            else:
                input_one = []
                features = line.strip().split(',')
                for i, v in enumerate(features):
                    v = self.scaler(v, feature_list, i)
                    input_one.append(v)
                input_list.append(input_one)
        if len(input_list) < self.n_temporal:
            for _ in range(self.n_temporal - len(input_list)):
                padvs = [0 for i in range(self.n_features + 1)]
                input_list = [padvs] + input_list
        input_list = np.array(input_list, dtype=np.float32)
        return input_list



    
    def __getitem__(self, idx):
        adm_id = self.adm_ids[idx]
        x_demo = np.array(self.demo_dict[adm_id], dtype=np.float32)

        # x_notes = self.notes_dict[adm_id]
        # x_notes = np.array([self.doc2vec.infer_vector(text2words(x_note)) for x_note in x_notes])
        # x_notes_max = np.max(x_notes, axis=0)
        # x_notes_mean = np.mean(x_notes, axis=0)
        # x_notes_min = np.min(x_notes, axis=0)
        # x_notes = np.hstack([x_notes_max, x_notes_mean, x_notes_min])

        x_notes = np.array([1])
        x_temporal = self.get_temporal(adm_id)
        x_temporal = x_temporal[:, 1:] # remove time


        y = np.array([int(label) for label in self.label_dict[adm_id]], dtype=np.float32)
        # print(x_demo, y)
        return torch.from_numpy(x_demo), torch.from_numpy(x_notes), torch.from_numpy(x_temporal), torch.from_numpy(y)

    def __len__(self):
        return len(self.adm_ids)
