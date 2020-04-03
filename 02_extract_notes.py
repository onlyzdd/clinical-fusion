import pandas as pd
import numpy as np

from utils import clean_text

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--firstday', default=False, action='store_true', help='extract first day notes')
    args = parser.parse_args()
    return args


def extract_early(df_notes, early_categories):
    '''Extract first 24 hours notes'''
    df_early = df_notes[df_notes['category'].isin(early_categories)]
    df_early['hr'] = (df_early['charttime'] - df_early['admittime']) / np.timedelta64(1, 'h')
    df_early = df_early[df_early['hr'] <= 24]
    # df_early = df_early.groupby('hadm_id').head(12).reset_index()
    df_early = df_early.sort_values(['hadm_id', 'hr'])
    df_early['text'] = df_early['text'].apply(clean_text)
    df_early[['hadm_id', 'hr', 'category', 'text']].to_csv('./data/processed/earlynotes.csv', index=None)


def extract_first(df_notes, early_categories):
    '''Extract first 24 notes'''
    df_early = df_notes[df_notes['category'].isin(early_categories)]
    df_early['hr'] = (df_early['charttime'] - df_early['admittime']) / np.timedelta64(1, 'h')
    df_early = df_early.groupby('hadm_id').head(24).reset_index()
    df_early = df_early.sort_values(['hadm_id', 'hr'])
    df_early['text'] = df_early['text'].apply(clean_text)
    df_early[['hadm_id', 'hr', 'category', 'text']].to_csv('./data/processed/earlynotes.csv', index=None)


if __name__ == '__main__':
    args = parse_args()
    print('Reading data...')
    early_categories = ['Nursing', 'Nursing/other', 'Physician ', 'Radiology']
    df_notes = pd.read_csv('./data/mimic/NOTEEVENTS.csv', parse_dates=['CHARTTIME'])
    df_notes.columns = map(str.lower, df_notes.columns)
    df_notes = df_notes[df_notes['iserror'].isnull()]
    df_notes = df_notes[~df_notes['hadm_id'].isnull()]
    df_notes = df_notes[~df_notes['charttime'].isnull()]
    
    df_adm = pd.read_csv('./data/mimic/adm_details.csv', parse_dates=['admittime'])
    df_notes = df_notes.merge(df_adm, on='hadm_id', how='left')

    if args.firstday:
        print('Extracting first day notes...')
        extract_early(df_notes, early_categories)
    else:
        print('Extracting first 24 notes...')
        extract_first(df_notes, early_categories)

