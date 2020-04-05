import pandas as pd
import numpy as np


if __name__ == "__main__":
    df_static = pd.read_csv('./data/processed/demo.csv')
    df_features = pd.read_csv('./data/processed/features.csv')
    df_notes = pd.read_csv('./data/processed/earlynotes.csv')
    df_icd = pd.read_csv('./data/processed/labels_icd.csv')
    df_notes = df_notes[~df_notes['text'].isnull()]
    adm_ids = df_static['hadm_id'].tolist()

    adm_ids = np.intersect1d(adm_ids, df_features['hadm_id'].unique().tolist())
    adm_ids = np.intersect1d(adm_ids, df_notes['hadm_id'].unique().tolist())
    adm_ids = np.intersect1d(adm_ids, df_icd['hadm_id'].unique().tolist())

    df_static[df_static['hadm_id'].isin(adm_ids)].to_csv('./data/processed/demo.csv', index=None)
    df_features[df_features['hadm_id'].isin(adm_ids)].to_csv('./data/processed/features.csv', index=None)
    df_notes[df_notes['hadm_id'].isin(adm_ids)].to_csv('./data/processed/earlynotes.csv', index=None)

    for task in ('mortality', 'readmit', 'los'):
        df = pd.read_csv('./data/processed/{}.csv'.format(task))
        df[df['hadm_id'].isin(adm_ids)].to_csv('./data/processed/{}.csv'.format(task), index=None)
    
    df = pd.read_csv('./data/processed/los.csv')
    df['llos'] = (df['los'] > 7).astype(int)
    df[['hadm_id', 'llos']].to_csv('./data/processed/llos.csv', index=None)

    df_icd[df_icd['hadm_id'].isin(adm_ids)].to_csv('./data/processed/labels_icd.csv', index=None)
