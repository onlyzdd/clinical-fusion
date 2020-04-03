import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils import bin_age, convert_icd_group


if __name__ == '__main__':
    df_adm = pd.read_csv('./data/mimic/adm_details.csv', parse_dates=[
        'dob', 'dod', 'admittime', 'dischtime'])
    print('Total admissions:', len(df_adm))
    df_adm = df_adm[df_adm['has_chartevents_data'] == 1]
    df_adm['age'] = df_adm['admittime'].subtract(
        df_adm['dob']).dt.days / 365.242
    df_adm['los'] = (df_adm['dischtime'] - df_adm['admittime']) / np.timedelta64(1, 'D')
    df_adm = df_adm[df_adm['age'] >= 18]  # keep adults
    df_adm['age'] = df_adm['age'].apply(bin_age)
    print('After removing non-adults:', len(df_adm))
    df_adm = df_adm[df_adm['los'] >= 1]  # keep more than 1 day
    print('After removing less than 1 day:', len(df_adm))
    df_adm = df_adm.sort_values(['subject_id', 'admittime']).reset_index(drop=True)

    print('Processing patients demographics...')
    df_adm['marital_status'] = df_adm['marital_status'].fillna('Unknown')
    df_static = df_adm[['hadm_id', 'age', 'gender', 'admission_type', 'insurance',
            'marital_status', 'ethnicity']]
    df_static.to_csv('./data/processed/demo.csv', index=None)

    print('Collecting labels...')

    df_icd = pd.read_csv('./data/mimic/DIAGNOSES_ICD.csv')[['HADM_ID', 'ICD9_CODE']].dropna()
    df_icd.columns = map(str.lower, df_icd.columns)
    df_icd['icd9_code'] = df_icd['icd9_code'].apply(convert_icd_group)
    df_icd = df_icd.dropna().drop_duplicates().sort_values(['hadm_id', 'icd9_code'])
    for x in range(20):
        x += 1
        df_icd[f'{x}'] = (df_icd['icd9_code'] == x).astype(int)
    df_icd = df_icd.groupby('hadm_id').sum()
    df_icd = df_icd[df_icd.columns[1:]].reset_index()
    df_icd = df_icd[df_icd.hadm_id.isin(df_adm.hadm_id)]

    df_readmit = df_adm.copy()
    df_readmit['next_admittime'] = df_readmit.groupby(
        'subject_id')['admittime'].shift(-1)
    df_readmit['next_admission_type'] = df_readmit.groupby(
        'subject_id')['admission_type'].shift(-1)
    elective_rows = df_readmit['next_admission_type'] == 'ELECTIVE'
    df_readmit.loc[elective_rows, 'next_admittime'] = pd.NaT
    df_readmit.loc[elective_rows, 'next_admission_type'] = np.NaN
    df_readmit[['next_admittime', 'next_admission_type']] = df_readmit.groupby(
        ['subject_id'])[['next_admittime', 'next_admission_type']].fillna(method='bfill')
    df_readmit['days_next_admit'] = (
        df_readmit['next_admittime'] - df_readmit['dischtime']).dt.total_seconds() / (24 * 60 * 60)
    df_readmit['readmit'] = (
        df_readmit['days_next_admit'] < 30).astype('int')
    

    print('Done.')
    df_labels = df_adm[['hadm_id', 'los']]
    df_labels['mortality'] = df_adm['hospital_expire_flag']
    df_labels['readmit'] = df_readmit['readmit']

    df_labels[['hadm_id', 'los']].to_csv('./data/processed/los.csv', index=None)
    df_labels[['hadm_id', 'mortality']].to_csv('./data/processed/mortality.csv', index=None)
    df_labels[['hadm_id', 'readmit']].to_csv('./data/processed/readmit.csv', index=None)
    df_icd.to_csv('./data/processed/labels_icd.csv', index=None)


