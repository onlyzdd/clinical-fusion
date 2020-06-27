import pandas as pd
import numpy as np


def cal_demo():
    df_adm = pd.read_csv('data/mimic/adm_details.csv',
                         parse_dates=['admittime', 'dischtime', 'dob'])
    df_adm['age'] = df_adm['admittime'].subtract(
        df_adm['dob']).dt.days / 365.242
    df_adm['los'] = (df_adm['dischtime'] - df_adm['admittime']
                     ) / np.timedelta64(1, 'D')
    df_adm['gender'] = (df_adm['gender'] == 'M').astype(int)
    result = []
    for task in ['mortality', 'readmit', 'llos']:
        df = pd.read_csv('./data/processed/%s.csv' % task)
        df = df.merge(df_adm, on='hadm_id', how='left')
        for label in [0, 1]:
            df_part = df[df[task] == label]
            total = len(df_part)
            n_emergency = len(
                df_part[df_part['admission_type'] == 'EMERGENCY'])
            n_elective = len(df_part[df_part['admission_type'] == 'ELECTIVE'])
            n_urgent = len(df_part[df_part['admission_type'] == 'URGENT'])
            mean_age, std_age = df_part['age'].mean(), df_part['age'].std()
            mean_los, std_los = df_part['los'].mean(), df_part['los'].std()
            result.append([task, label, n_elective, n_emergency,
                           n_urgent, total, mean_age, std_age, mean_los, std_los])
    df_result = pd.DataFrame(result, columns=['task', 'label', 'elective', 'emergency',
                                              'urgent', 'total', 'age (mean)', 'age (std)', 'los (mean)', 'los (std)'])
    print(df_result)


def cal_temporal():
    df = pd.read_csv('data/processed/features.csv')
    df_result = df.describe().transpose()
    df_result['missing'] = df.isna().mean()
    print(df_result)


if __name__ == '__main__':
    cal_demo()
    cal_temporal()
