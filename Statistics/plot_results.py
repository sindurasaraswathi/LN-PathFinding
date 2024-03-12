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
from ordered_set import OrderedSet

plt.style.use('ggplot')
# f1 = pd.read_csv('/Users/ssarasw2/Desktop/LN pathfinding/LN-PathFinding/New_MP_results/LN_results_random_mp_new.csv')
# f2 = pd.read_csv('/Users/ssarasw2/Desktop/LN pathfinding/LN-PathFinding/New_MP_results/LN_results_random_mp_large.csv')
# f2 = f2[f2['Amount']>(10**6)]
# f3 = pd.read_csv('/Users/ssarasw2/Desktop/LN pathfinding/LN-PathFinding/New_MP_results/LN_results_random_mp_large_104.csv')
# f3 = f3[f3['Amount']>(10**5)]

# df = pd.concat([f1,f2,f3], axis=0)
# df = df.dropna()
# df = df.drop_duplicates()

df = pd.read_csv('/Users/ssarasw2/Desktop/LN pathfinding/LN-PathFinding/New_MP_results/LN_results_new_10k.csv')
df = df.dropna()
df = df.drop_duplicates()

#-------------------------------------------------------------------------------------------------
#seperate each field from the string
def extract_field(num, col):
    extract_list = []
    for i in df[col]:
        elem = ast.literal_eval(i)
        extract_list.append(elem[num])
    return extract_list

#-------------------------------------------------------------------------------------------------
#store values (fee, delay, pathlength, throughput) in df1 dataframe
df1 = pd.DataFrame()
df1['amount'] = df['Amount']
algo = ['LND', 'CLN','LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3']
for a in algo:
    df1[f'{a}fee'] = extract_field(1, a)
    df1[f'{a}dly'] = extract_field(2, a)
    df1[f'{a}pthlnt'] = extract_field(3, a)
    df1[f'{a}tp'] = extract_field(4, a)

#-------------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------------------------
#scale df to 100
sdf = pd.DataFrame(srate)
fdf = pd.DataFrame(frate)
for a in algo:
    sdf[a] = sdf[a]*100/sdf['Amount']
sdf['Amount'] = sdf['Amount']*100/sdf['Amount']
# sdf['Bins'] = [f'{i}-{i+1}' for i in range(8)]
sdf.plot(kind = 'bar')
plt.ylabel('Count')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

sdf['Actual Amount'] = amt_bins
# print(sdf.to_latex(index=False))
print('\n', tabulate(sdf, headers = 'keys', tablefmt = 'psql',showindex=True))
print('\nSuccess Rate:\n\n', tabulate(srate, headers = 'keys', tablefmt = 'psql',showindex=False))
# print('\nFailure Rate:\n\n', tabulate(frate, headers = 'keys', tablefmt = 'psql',showindex=False))

#-------------------------------------------------------------------------------------------------
def df_plot(data, amt_bins, algo, title, xlabel, ylabel):
    df_temp = pd.DataFrame(data)
    ratio_df = pd.DataFrame()
    for a in algo:
        ratio_df[a] = df_temp[a]/df_temp['Amount']
    ratio_df.plot()
    # df_temp.plot(kind='bar')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

df_plot(srate, amt_bins, algo, 'Success Rate', 'Amount bins', 'Ratio')
# df_plot(frate, amt_bins, algo, 'Failure Rate', 'Amount bins', 'Ratio')
pd.DataFrame(srate).plot(kind='bar')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Success Rate')
plt.xlabel('Amount bins')
plt.ylabel('Count')
plt.show()

#-------------------------------------------------------------------------------------------------
#graph types
def plot_graph(x, y, kind, xlog, ylog, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=plt.figaspect(1/3))
    if kind == 'scatter':
        ax.scatter(x,y)
    elif kind == 'box':
        ax.boxplot(x, showfliers=False, whis = 0.0, showmeans=True)
    plt.title(title)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def fee_df(val, name, step):
    end = 8
    fee_list = []
    count = []
    fee_med = []
    amount_list = []
    i = 0
    while i<end:
        if i==0:
            lrange = 0
        else:
            lrange = 10**i
        i = i+step
        if i>end:
            i=end
        rrange = 10**(i)
        count.append((lrange,rrange))
        data = val[(val['amount']>lrange) & (val['amount']<=rrange)]
        fee_med.append(data[name].median())
        fee_list.append(list(data[name]))
        amount_list.append(list(data['amount']))
    return fee_list, fee_med, amount_list
    
       
for a in algo:
    pdf = pd.DataFrame(columns=['avg amount', 'avg path length', 'avg median fee'])
    grp_val = df1[[f'{a}fee', 'amount']].sort_values('amount').groupby('amount')
    pth_grp = df1[[f'{a}pthlnt', 'amount']].sort_values('amount').groupby('amount').mean()
    for fltr in ['Success', 'Overall']:
        if fltr == 'Success':
            val = df1[df1[f'{a}tp']=='Success'][[f'{a}fee', 'amount']]
        else:
            val = df1[[f'{a}fee', 'amount']]
        name = f'{a}fee'
        step = 1
        fee_list, fee_med, amount_list = fee_df(val, name, step)
        # plot_graph(fee_list, 0, 'box', False, True, f'{a} Fee vs Amount ({fltr})', 'Amount', 'Fee')
        # plot_graph(range(len(fee_med)), fee_med,'scatter', False, True, f'{a} Median Fee vs Amount ({fltr})', 'Amount', 'Fee')
        
    val = []
    key = []
    for i, j in grp_val:
        if i%50 == 0 and i<=1000 and i>=100:
            val.append(list(j[f'{a}fee']))
            key.append(i)
                
                
         
    for j in range(8):
        pval = []
        pkey = []
        fval = []
        for i in pth_grp.index:
            if i>10**j and i<=10**(j+1):
                pkey.append(i)
                pval.append(pth_grp.loc[i][f'{a}pthlnt'])
                fval.append(grp_val.get_group(i)[f'{a}fee'].median())
        pdf.loc[j] = [mean(pkey), mean(pval), mean(fval)]
    print(a,'\n', tabulate(pdf, headers = 'keys', tablefmt = 'psql',showindex=True))
    
        # for i in range(len(amount_list)):
        #     amt_list = list(OrderedSet(amount_list[i]))
        #     key.append(amt_list)
        #     temp = []
        #     for j in amt_list:
        #         temp.append(list(grp_val.get_group(j)[f'{a}fee']))
        #     val.append(temp)
# plot_graph(val, 0, 'box', False, False, f'{a} Fee vs Amount ({fltr})', 'Amount', 'Fee')

plt.boxplot(val, showfliers=False)
plt.xticks(range(1,len(key)+1), key)
plt.show()

#-------------------------------------------------------------------------------------------------
def sns_plot(data, kind, xlog, ylog, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=plt.figaspect(1/3))
    if kind != 'hist':
        sns.displot(data, kind=kind)
    else:
        ax.hist(data)
        plt.legend(algo)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

data = df1[['LNDpthlnt', 'LDKpthlnt', 'CLNpthlnt', 'Eclair_case1pthlnt', 'Eclair_case2pthlnt', 'Eclair_case3pthlnt']]
# for a in algo:
#     data = df1[df1[f'{a}tp']=='Success'][f'{a}pthlnt']
#     path_avg.append(data.mean())
# print(path_avg)
sns_plot(data, 'hist', False, False, 'Path length', '', '')
sns_plot(data, 'kde', False, False, 'Path length (KDE)', '', '')
# print(data.mean())


data = df1[['LNDdly', 'LDKdly', 'CLNdly', 'Eclair_case1dly', 'Eclair_case2dly', 'Eclair_case3dly']]
print(data.mean())
data=df1[['LNDfee', 'LDKfee', 'CLNfee', 'Eclair_case1fee', 'Eclair_case2fee', 'Eclair_case3fee']]
print(data.mean())


#-------------------------------------------------------------------------------------------------
# def save_df(data, filename):
#     df_temp = pd.DataFrame(data) #dict data
#     df_temp.to_csv(filename)
    
# save_df(srate, 'success_count.csv')
# save_df(frate, 'Failure_count.csv')
# df1.to_csv('result_data.csv')


