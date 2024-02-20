#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:47:20 2024

@author: ssarasw2
"""

import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('New_results/LN_results_random.csv')
df = df.dropna()

def extract_field(num, col):
    extract_list = []
    for i in df[col]:
        elem = ast.literal_eval(i)
        extract_list.append(elem[num])
    return extract_list

df1 = pd.DataFrame()
df1['amount'] = df['Amount']
algo = {'LND', 'CLN','LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3'}
for a in algo:
    df1[f'{a}fee'] = extract_field(1, a)
    df1[f'{a}dly'] = extract_field(2, a)
    df1[f'{a}pthlnt'] = extract_field(3, a)
    df1[f'{a}tp'] = extract_field(4, a)


srate = {}
frate = {}
start = 0
end = 8
step = 1
for a in algo:
    slist = []
    flist = []
    amt_bins = []
    for i in range(start, end, step):
        if i == 0:
            lrange = 0
        else:
            lrange = 10**i
        rrange = 10**(i+1)
        data = df1[(df1['amount']>lrange) & (df1['amount']<=rrange)]
        amt_bins.append(len(data))
        slist.append(len(data[data[f'{a}tp'] == 'Success']))
        flist.append(len(data[data[f'{a}tp'] == 'Failure']))
    
    srate[a] = slist
    frate[a] = flist
    
print(srate)
# sns.displot(data=df1[['LNDfee', 'LDKfee', 'CLNfee', 'Eclair_case1fee', 'Eclair_case2fee', 'Eclair_case3fee']], kind='ecdf')
# plt.xscale('log')
# plt.show()

# sns.displot(data=df1[['LNDdly', 'LDKdly', 'CLNdly', 'Eclair_case1dly', 'Eclair_case2dly', 'Eclair_case3dly']], kind='ecdf')
# plt.xscale('log')

# plt.show()

# sns.displot(data=df1[['LNDpthlnt', 'LDKpthlnt', 'CLNpthlnt', 'Eclair_case1pthlnt', 'Eclair_case2pthlnt', 'Eclair_case3pthlnt']], kind='ecdf')
# plt.xscale('log')
# plt.show()


df1 = pd.DataFrame(frate)
df1['Amount'] = amt_bins

ratio_df = pd.DataFrame()
for a in algo:
    ratio_df[a] = df1[a]/df1['Amount']

ratio_df.plot()
plt.show()
# # df1.to_csv('filtered_data_fail.csv')

# df1.plot(kind='line')
# box_pts = 10
# box = np.ones(box_pts)/box_pts
# x = np.arange(end)
# plt.bar(x, np.convolve(df1['LND'], box, mode='same'))
# plt.show()


