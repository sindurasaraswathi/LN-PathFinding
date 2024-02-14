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

df = pd.read_csv('New_results/LN_results_random.csv')
df = df.dropna()

def extract_field(num, col):
    extract_list = []
    for i in df[col]:
        elem = ast.literal_eval(i)
        extract_list.append(elem[num])
    return extract_list

lnd_fee = extract_field(1, 'LND')
cln_fee = extract_field(1, 'CLN')
ldk_fee = extract_field(1, 'LDK')
ec1_fee = extract_field(1, 'Eclair_case1')
ec2_fee = extract_field(1, 'Eclair_case2')
ec3_fee = extract_field(1, 'Eclair_case3')

lnd_dly = extract_field(2, 'LND')
cln_dly = extract_field(2, 'CLN')
ldk_dly = extract_field(2, 'LDK')
ec1_dly = extract_field(2, 'Eclair_case1')
ec2_dly = extract_field(2, 'Eclair_case2')
ec3_dly = extract_field(2, 'Eclair_case3')

lnd_lnt = extract_field(3, 'LND')
cln_lnt = extract_field(3, 'CLN')
ldk_lnt = extract_field(3, 'LDK')
ec1_lnt = extract_field(3, 'Eclair_case1')
ec2_lnt = extract_field(3, 'Eclair_case2')
ec3_lnt = extract_field(3, 'Eclair_case3')

lnd_srate = extract_field(4, 'LND')
cln_srate = extract_field(4, 'CLN')
ldk_srate = extract_field(4, 'LDK')
ec1_srate = extract_field(4, 'Eclair_case1')
ec2_srate = extract_field(4, 'Eclair_case2')
ec3_srate = extract_field(4, 'Eclair_case3')

dly_df = pd.DataFrame({'LND_dly':lnd_dly,'CLN_dly':cln_dly, 'LDK_dly':ldk_dly, 'Eclair_1_dly':ec1_dly, 'Eclair_2_dly':ec2_dly, 'Eclair_3_dly':ec3_dly})
fee_df = pd.DataFrame({'LND':lnd_fee,'CLN':cln_fee, 'LDK':ldk_fee, 'Eclair_1':ec1_fee, 'Eclair_2':ec2_fee, 'Eclair_3':ec3_fee})
lnt_df = pd.DataFrame({'LND':lnd_lnt,'CLN':cln_lnt, 'LDK':ldk_lnt, 'Eclair_1':ec1_lnt, 'Eclair_2':ec2_lnt, 'Eclair_3':ec3_lnt})
srate_df = pd.DataFrame({'LND':lnd_srate,'CLN':cln_srate, 'LDK':ldk_srate, 'Eclair_1':ec1_srate, 'Eclair_2':ec2_srate, 'Eclair_3':ec3_srate})

amounts = df['Amount']


data = pd.concat([dly_df, fee_df], axis=1)

sns.displot(data=data,kind='ecdf', log_scale=True)
plt.xlabel('Fee')
plt.ylabel('ECDF')
plt.show()

sns.displot(data=dly_df,kind='ecdf', log_scale=True)
plt.xlabel('Delay')
plt.ylabel('ECDF')
plt.show()

sns.displot(data=lnt_df,kind='ecdf', log_scale=True)
plt.xlabel('Path Length')
plt.ylabel('ECDF')
plt.show()

# sns.catplot(data=lnd_srate, x=)
# plt.xlabel('Success Rate')
plt.show()





