import pandas as pd
from scipy.stats import ttest_ind


def cal_pvalue(data1, data2):
    t, p = ttest_ind(data1, data2)
    return p


if __name__ == "__main__":
    df_baselines = pd.read_csv('./results/baselines.csv')
    df_deep = pd.read_csv('./results/deep.csv')
    df = pd.concat([df_baselines, df_deep])
    for task in ['mortality', 'llos', 'readmit']:
        data = {}
        for model in ['lr', 'rf', 'cnn', 'lstm']:
            for inputs in [3, 4, 7]:
                df_tmp = df[(df['task'] == task) & (
                    df['model'] == model) & (df['inputs'] == inputs)]
                data[f'{model}-{inputs}'] = df_tmp['auc'].tolist()
        print(',' + ','.join(data.keys()))
        for i in data.keys():
            result = [i]
            for j in data.keys():
                pvalue = cal_pvalue(data[i], data[j])
                if pvalue < 0.001:
                    result.append('< 0.001')
                else:
                    result.append('%.4f' % pvalue)
            print(','.join(result))
