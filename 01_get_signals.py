import pandas as pd
import numpy as np


def get_signals(start_hr, end_hr):
    df_adm = pd.read_csv('./data/mimic/adm_details.csv',
                         parse_dates=['admittime'])
    adm_ids = df_adm.hadm_id.tolist()
    for signal in ['vital', 'lab']:
        df = pd.read_csv(
            './data/mimic/pivoted_{}.csv'.format(signal), parse_dates=['charttime'])
        df = df.merge(df_adm[['hadm_id', 'admittime']], on='hadm_id')
        df = df[df.hadm_id.isin(adm_ids)]
        df['hr'] = (df.charttime - df.admittime) / np.timedelta64(1, 'h')
        df = df[(df.hr <= end_hr) & (df.hr >= start_hr)]
        df = df.set_index('hadm_id').groupby('hadm_id').resample(
            'H', on='charttime').mean().reset_index()
        df.to_csv('./data/mimic/{}.csv'.format(signal), index=None)
    df = pd.read_csv('./data/mimic/vital.csv', parse_dates=['charttime'])[
        ['hadm_id', 'charttime', 'heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2']]
    df_lab = pd.read_csv('./data/mimic/lab.csv',
                         parse_dates=['charttime'])
    df = df.merge(df_lab, on=['hadm_id', 'charttime'], how='outer')
    df = df.merge(df_adm[['hadm_id', 'admittime']], on='hadm_id')
    df['charttime'] = ((df.charttime - df.admittime) / np.timedelta64(1, 'h'))
    df['charttime'] = df['charttime'].apply(np.ceil) + 1
    df = df[(df.charttime <= end_hr) & (df.charttime >= start_hr)]
    df = df.sort_values(['hadm_id', 'charttime'])
    df['charttime'] = df['charttime'].map(lambda x: int(x))
    df = df.drop(['admittime', 'hr'], axis=1)
    na_thres = 3
    df = df.dropna(thresh=na_thres)
    df.to_csv('./data/processed/features.csv', index=None)


if __name__ == '__main__':
    get_signals(1, 24)
