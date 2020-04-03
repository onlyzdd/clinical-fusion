import pandas as pd
import numpy as np


def cal_icd_ratio():
    df = pd.read_csv('data/processed/labels_icd.csv')
    print(df.mean())


def cal_demo():
    df_adm = pd.read_csv('data/mimic/adm_details.csv', parse_dates=['admittime', 'dischtime', 'dob'])
    df_adm['age'] = df_adm['admittime'].subtract(
        df_adm['dob']).dt.days / 365.242
    df_adm['los'] = (df_adm['dischtime'] - df_adm['admittime']) / np.timedelta64(1, 'D')
    df_adm['gender'] = (df_adm['gender'] == 'M').astype(int)
    for task in ['mortality', 'readmit', 'los_bin']:
        df = pd.read_csv('./data/processed/%s.csv' % task)
        df = df.merge(df_adm, on='hadm_id', how='left')
        for label in [0, 1]:
            print('%s %d:' % (task, label))
            df_part = df[df[task] == label]
            print(df_part['admission_type'].value_counts())
            print(len(df_part), df_part['age'].mean(), df_part['los'].mean(), df_part['gender'].sum())


if __name__ == '__main__':
    # cal_icd_ratio()
    cal_demo()