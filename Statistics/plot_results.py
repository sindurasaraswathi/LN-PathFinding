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
from statistics import mean, mode, median
from tabulate import tabulate

plt.style.use('ggplot')
df = pd.read_csv('/Users/ssarasw2/Desktop/LN pathfinding/LN-PathFinding/New_MP_results/LN_results_random_mp_large_104.csv')
df = df.dropna()

#seperate each field from the string
def extract_field(num, col):
    extract_list = []
    for i in df[col]:
        elem = ast.literal_eval(i)
        extract_list.append(elem[num])
    return extract_list

#store values (fee, delay, pathlength, throughput) in df1 dataframe
df1 = pd.DataFrame()
df1['amount'] = df['Amount']
algo = ['LND', 'CLN','LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3']
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
#calculate the count of success and failure in the bin of amount
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

    
srate['Amount'] = amt_bins
frate['Amount'] = amt_bins

sdf = pd.DataFrame(srate)
fdf = pd.DataFrame(frate)


for a in algo:
    sdf[a] = sdf[a]*100/sdf['Amount']
sdf['Amount'] = sdf['Amount']*100/sdf['Amount']
sdf['Bins'] = [f'{i}-{i+1}' for i in range(8)]
sdf.plot(kind = 'bar')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# print(sdf.to_latex(index=False))
print('\n', tabulate(sdf, headers = 'keys', tablefmt = 'psql',showindex=False))
print('\nSuccess Rate:\n\n', tabulate(srate, headers = 'keys', tablefmt = 'psql',showindex=False))
print('\nFailure Rate:\n\n', tabulate(frate, headers = 'keys', tablefmt = 'psql',showindex=False))


def df_plot(data, amt_bins, algo, title, xlabel, ylabel):
    df_temp = pd.DataFrame(data)
    ratio_df = pd.DataFrame()
    for a in algo:
        ratio_df[a] = df_temp[a]/df_temp['Amount']
    ratio_df.plot()
    df_temp.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


df_plot(srate, amt_bins, algo, 'Success Rate', 'Amount bins', 'Ratio')
df_plot(frate, amt_bins, algo, 'Failure Rate', 'Amount bins', 'Ratio')


def save_df(data, filename):
    df_temp = pd.DataFrame(data) #dict data
    df_temp.to_csv(filename)
    
save_df(srate, 'success_count.csv')
save_df(frate, 'Failure_count.csv')
df1.to_csv('result_data.csv')


#bar graphs

def plot_graph(x, y, kind, xlog, ylog, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=plt.figaspect(1/3))
    if kind == 'scatter':
        ax.scatter(x,y)
    elif kind == 'hist':
        plt.hist(x, bins=y)
    elif kind == 'bar':
        plt.bar(x,y)
        
    plt.title(title)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
new_df = pd.DataFrame()
for a in algo:    
    plot_graph(df1['amount'], df1[f'{a}fee'], 'scatter', True, True, f'{a} Fee vs Amount', 'Amount (log scale)', 'Fee (log scale)')
    # plot_graph(df1[f'{a}fee'], 1000, 'hist', True, True, f'{a} Fee vs Amount', 'Amount (log scale)', 'Fee (log scale)')
    new_df[a] = df1[f'{a}fee']
plt.show()

def sns_plot(data, kind, xlog, ylog, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=plt.figaspect(1/3))
    # ax.hist(data)
    sns.displot(data, kind=kind)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(algo)
    plt.show()


# data = df1[['LNDdly', 'LDKdly', 'CLNdly', 'Eclair_case1dly', 'Eclair_case2dly', 'Eclair_case3dly']]
# sns_plot(data, 'ecdf', True, False, 'Delay ECDF', '', 'Proportion')


data = df1[['LNDpthlnt', 'LDKpthlnt', 'CLNpthlnt', 'Eclair_case1pthlnt', 'Eclair_case2pthlnt', 'Eclair_case3pthlnt']]
# sns_plot(data, 'hist', False, False, 'Path length', '', '')
sns_plot(data, 'kde', False, False, 'Path length (KDE)', '', '')



# sns.displot(data=df1[['LNDfee', 'LDKfee', 'CLNfee', 'Eclair_case1fee', 'Eclair_case2fee', 'Eclair_case3fee']], kind='ecdf')
# plt.xscale('log')
# plt.show()

# sns.displot(data=df1[['LNDdly', 'LDKdly', 'CLNdly', 'Eclair_case1dly', 'Eclair_case2dly', 'Eclair_case3dly']], kind='ecdf')


# sns.displot(data=df1[['LNDpthlnt', 'LDKpthlnt', 'CLNpthlnt', 'Eclair_case1pthlnt', 'Eclair_case2pthlnt', 'Eclair_case3pthlnt']], kind='ecdf')
# plt.xscale('log')
# plt.show()


