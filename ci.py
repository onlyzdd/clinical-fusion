import pandas as pd
import numpy as np
from scipy.stats import sem, t


def cal_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    start, end = t.interval(confidence, n - 1, m, std_err)
    return (end - start) / 2


if __name__ == "__main__":
    df_baselines = pd.read_csv('./results/baselines.csv')
    df_deep = pd.read_csv('./results/deep.csv')
    df = pd.concat([df_baselines, df_deep])
    i = 0
    print('task,model,inputs,auroc')
    for task in ['mortality', 'llos', 'readmit']:
        for model in ['lr', 'rf', 'cnn', 'lstm']:
            for inputs in [3, 4, 7]:
                df_tmp = df[(df['task'] == task) & (
                    df['model'] == model) & (df['inputs'] == inputs)]
                data = df_tmp['auc'].tolist()
                ci = cal_ci(data)
                auroc = np.mean(data)
                start, end = auroc - ci, auroc + ci
                i += 1
                print('%s,%s,%d,"%.3f (%.3f, %.3f)"' %
                      (task, model, inputs, auroc, start, end))
