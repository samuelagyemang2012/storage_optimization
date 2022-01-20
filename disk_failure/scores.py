import os
import pandas as pd
import numpy as np

base_path = "../disk_failure/results/xgb/"


def get_avgs(path):
    files = os.listdir(path)
    dfs = []
    for f in files:
        full_path = path + f
        dx = pd.read_csv(full_path)
        dfs.append(dx)

    df = pd.concat(dfs)
    auc = df[df['Eval AUC'] <= 1]['Eval AUC'].to_list()

    s = sum(auc) / len(auc)
    return round(s, 2)


def get_max(path):
    files = os.listdir(path)
    dfs = []
    for f in files:
        full_path = path + f
        dx = pd.read_csv(full_path)
        dfs.append(dx)

    df = pd.concat(dfs)
    adx = df[df['Feature Type'] == 'All features']
    pdx = df[df['Feature Type'] == 'Pearson features']
    sdx = df[df['Feature Type'] == 'Spearman features']

    a = adx[adx['Eval AUC'] <= 1]['Eval AUC'].to_list()
    p = pdx[pdx['Eval AUC'] <= 1]['Eval AUC'].to_list()
    s = sdx[sdx['Eval AUC'] <= 1]['Eval AUC'].to_list()

    a_ = max(a)
    p_ = max(p)
    s_ = max(s)

    return a_, p_, s_


print('XGB')
ax, px, sx = get_max('../disk_failure/results/xgb/')
print(ax, px, sx)
print('')

print('D Tree')
at, pt, st = get_max('../disk_failure/results/tree/')
print(at, pt, st)
print('')

print('Random Forest')
ar, pr, sr = get_max('../disk_failure/results/rf/')
print(ar, pr, sr)
