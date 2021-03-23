from matplotlib.pyplot import plot
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


pd.options.display.float_format = "{:,.1f}".format

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


def cal_task_temporal():
    df_temporal = pd.read_csv('data/processed/features.csv')
    for task in ['mortality', 'readmit', 'llos']:
        df_label = pd.read_csv('data/processed/%s.csv' % task)
        for label in [0, 1]:
            df = df_temporal[df_temporal['hadm_id'].isin(df_label[df_label[task] == label]['hadm_id'])]
            df = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()
            print(task, label)
            print(df)


def plot_los():
    df = pd.read_csv('data/processed/los.csv')
    plt.figure(figsize=(8, 4))
    plt.hist(df['los'], bins=60)
    plt.axvline(x=7, color='r', linestyle='-')
    plt.xlabel('Length of stay (day)')
    plt.ylabel('# of patients')
    plt.title('Length of stay distribution of the processed MIMIC-III cohort    ')
    plt.savefig('imgs/los_dist.png')


def plot_temporal():
    df = pd.read_csv('data/processed/features.csv')
    nrows, ncols = 4, 7
    # plt.figure(figsize=(28, 12))
    plt.clf()
    fig, axs = plt.subplots(nrows, ncols)
    cols = df.columns[2:]
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j < len(cols):
                print(j)
                col = cols[i * ncols + j]
                axs[i, j].hist(df[col], bins=20)
                axs[i, j].title.set_text(col)
    plt.savefig('imgs/temporal.png')


if __name__ == '__main__':
    # cal_demo()
    # cal_temporal()
    # cal_task_temporal()
    # plot_los()
    plot_temporal()
