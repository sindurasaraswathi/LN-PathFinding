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

# plt.style.use('ggplot')

res = ['10k', 'W_W', 'W_F', 'W_P', 'F_W', 'F_F', 'F_P', 'P_W', 'P_F', 'P_P']
cols = ['Index', 'LND1', 'LND2', 'CLN', 'LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3']
dff = pd.DataFrame(columns=cols)
dff['Index'] = res
dff.set_index('Index', inplace=True)

dff1 = pd.DataFrame(columns=cols)
dff1['Index'] = res
dff1.set_index('Index', inplace=True)

dff2 = pd.DataFrame(columns=cols)
dff2['Index'] = res
dff2.set_index('Index', inplace=True)

final_df = pd.DataFrame()

df_node = pd.DataFrame()


for file in res:
    df = pd.read_csv(f'C:/Users/sindu/Work/UNCC Research/GIT_LN/LN-PathFinding/New_results/LN_results_{file}.csv')
    df = df.fillna("[[],0,0,0,'Failure']")
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
    algo = list(df.columns[3:])
    for a in algo:
        df1[f'{a}fee'] = extract_field(1, a)
        df1[f'{a}dly'] = extract_field(2, a)
        df1[f'{a}pthlnt'] = extract_field(3, a)
        df1[f'{a}tp'] = extract_field(4, a)
    
    color = ['#1f77b4',  # Blue
                   '#ff7f0e',  # Orange
                   '#2ca02c',  # Green
                   '#d62728',  # Red
                   '#9467bd',  # Purple
                   '#8c564b',  # Brown
                   '#e377c2',  # Pink
                   '#7f7f7f',  # Gray
                   '#bcbd22',  # Yellow
                   '#17becf']  # Cyan
    #-------------------------------------------------------------------------------------------------
    srate = {}
    start = 0
    if 'F' in file:
        end = 6 #6 for fair
    else:
        end = 8
        
    step = 1
    #calculate the count of success and failure in the bin of amount
    for a in algo:
        slist = []
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
        srate[a] = slist
    srate['Txn Count'] = amt_bins
    
    #-------------------------------------------------------------------------------------------------
    #scale df to 100
    sdf = pd.DataFrame(srate)
    for a in algo:
        sdf[a] = sdf[a]*100/sdf['Txn Count']
    sdf['Txn Count'] = sdf['Txn Count']*100/sdf['Txn Count']
    # sdf['Bins'] = [f'{i}-{i+1}' for i in range(8)]
    sdf[sdf.columns[0:-1]].plot(kind='bar', color=color) #bar plot
    plt.xlabel('Amount bins')
    plt.ylabel('percentage')
    plt.title('Success Rate percentage')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    sdf['Actual Txn Count'] = amt_bins
    # print('\n', tabulate(sdf, headers = 'keys', tablefmt = 'psql',showindex=True))
    # print('\nSuccess Rate:\n\n', tabulate(srate, headers = 'keys', tablefmt = 'psql',showindex=False))
    # print('\nFailure Rate:\n\n', tabulate(frate, headers = 'keys', tablefmt = 'psql',showindex=False))
    
    #-------------------------------------------------------------------------------------------------
    # plot data frame (line graph), pass dict to the function
    def df_plot(data, amt_bins, algo, title, xlabel, ylabel):
        df_temp = pd.DataFrame(data)
        ratio_df = pd.DataFrame()
        for a in algo:
            ratio_df[a] = df_temp[a]/df_temp['Txn Count']
        ratio_df.plot(color=color)
        # df_temp.plot(kind='bar')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.savefig(f'C:/Users/sindu/OneDrive/Desktop/temp/successrate_{file}.png')
        plt.show()

    
    df_plot(srate, amt_bins, algo, 'Success Rate', 'Amount bins', 'Ratio') 
    
    
    pd.DataFrame(srate).plot(kind='bar', color=color) #bar
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
            ax.boxplot(x, showfliers=False, whis = 0.0, showmeans=False)
        plt.title(title)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.savefig(f'C:/Users/sindu/OneDrive/Desktop/temp/{title}_{file}.png')
        plt.show()

        
    #find median fee in the amount bin, return list of list of fees in bins, list of median fee   
    def fee_df(val, name, step):
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
            fee_med.append(((data[name]+0.00000001)/data['amount']).median()*100)
            fee_list.append(list((data[name]+0.00000001)/data['amount']))
            # fee_med.append((data[name]+0.00000001).median())
            # fee_list.append(list(data[name]+0.00000001))
            amount_list.append(list(data['amount']))
        return fee_list, fee_med, amount_list
     
    #filter data based on success rate across variants
    #If success on 4 or more algo, apppend to sfee
    sfee = pd.DataFrame(columns=df1.columns)
    for i in df1.index:
        c = 0
        row = df1.loc[i]
        for a in algo:
            if row[f'{a}tp'] == 'Success':
                c+=1
        if c>3:
            sfee =  pd.concat([sfee, pd.DataFrame([row])], ignore_index=True)  
           
             
    # save_df = pd.DataFrame(index=['Weighted avg median fee', 'WAvg path length', 'WAvg Delay', 'avg_path', 'Avg delay'])
    
    temp_df = pd.DataFrame()
    pdf = pd.DataFrame(columns=algo)
    print(file)
    for a in algo:
        grp_val = sfee[[f'{a}fee', 'amount']].sort_values('amount').groupby('amount')
        pth_grp = sfee[[f'{a}pthlnt', 'amount']].sort_values('amount').groupby('amount').mean()
        dly_grp = sfee[[f'{a}dly', 'amount']].sort_values('amount').groupby('amount').mean()
        val = sfee[[f'{a}fee', 'amount']]
        name = f'{a}fee'
        step = 1
        fee_list, fee_med, amount_list = fee_df(val, name, step)
        # plot_graph(fee_list, 0, 'box', False, True, f'{a} Fee', 'Amount', 'Fee')
        # plot_graph(range(len(fee_med)), fee_med,'scatter', False, True, f'{a} Median Fee', 'Amount', 'Fee')
            
        w_sum = 0 
        p_sum = 0
        d_sum = 0
        total_weight = 0   
        fee_mega = []                       
        for j in range(end):
            fee_mega = fee_mega+fee_list[j] #overall fee ratio list
        pdf[a] = fee_med
        
        #overall stat in a file
        dff.loc[file][a] = median(fee_mega)*100
        dff1.loc[file][a] = sfee[[f'{a}pthlnt']].mean()[0]
        dff2.loc[file][a] = sfee[[f'{a}dly']].mean()[0]
                
        
    #     temp_df[a] = [i for i in range(end+1)]
    #     temp_df = pd.concat([temp_df, pdf], axis=1)
    
    
    # temp_df['file'] = [file for i in range(end+1)]
    # final_df = pd.concat([final_df, temp_df], ignore_index=True)
    
    #bin wise stat in a file 
    sdf['file'] = file
    pdf['file'] = file
    for j in [1,3,5]:
        df_node = df_node.append(sdf.loc[j], ignore_index=True)
        final_df = final_df.append(pdf.loc[j], ignore_index=True)#fee

    
    # save_df.to_csv(f'{file}_stat2.csv')
    # if '10k' in file:
    #     sdf.to_csv(f'{file}_stat1.csv')
    
    #-------------------------------------------------------------------------------------------------
    def sns_plot(data, kind, xlog, ylog, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=plt.figaspect(1/3))
        if kind != 'hist':
            sns.displot(data, kind=kind)
        else:
            ax.hist(data, color=color[:7])
            plt.legend(algo)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.savefig(f'C:/Users/sindu/OneDrive/Desktop/temp/path_length_{file}.png')
        plt.show()

    data = sfee[[f'{a}pthlnt' for a in algo]]
    sns_plot(data, 'hist', False, False, 'Path Length', 'Path Length', 'Count')
    
