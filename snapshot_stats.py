#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:10:22 2024

@author: ssarasw2
"""
import pandas as pd
from statistics import mean, mode, median
from ordered_set import OrderedSet

df = pd.read_csv('LN_snapshot.csv')
is_multi = df["short_channel_id"].value_counts() > 1
df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]

avg_cap = 0
print("Maximum capacity = ", max(df['satoshis']))
print("Minimum capacity = ", min(df['satoshis']))
print("Average capacity = ", mean(df['satoshis']))
print("Most frequent capacity = ", mode(df['satoshis']))
print("Median capacity = ", median(df['satoshis']))

nodes_pubkey = list(OrderedSet(list(df['source']) + list(df['destination'])))
print("Number of nodes = ", len(nodes_pubkey))

node_num = {}
for i in range(len(nodes_pubkey)):
    pubkey = nodes_pubkey[i]
    node_num[pubkey] = i
    
src_count = df['source'].value_counts()
print("Average source channels = ", mean(src_count))
# src_count = src_count.drop_duplicates()# drop duplicate counts not duplicate nodes
# print("Average unique source channels = ", mean(src_count))

# dst_count = df['destination'].value_counts()
# print("Average dest channels = ", mean(dst_count))
# dst_count = dst_count.drop_duplicates() # drop duplicate counts not duplicate nodes
# print("Average unique dest channels = ", mean(dst_count))
# print("Top 5 well connected source nodes:")

# for i in src_count.index[:6]:
#     print("Node =", node_num[i], "no.of channels =", src_count[i])
    
# print("5 fairly connected source nodes:") #53:58 for unique set, wholeset -11:-6
# for i in src_count.index[53:58]:
#     print("Node =", node_num[i], "no.of channels =", src_count[i])
    
# print("5 poorly connected source nodes:")
# for i in src_count.index[-5:]:
#     print("Node =", node_num[i], "no.of channels =", src_count[i])
    
    
# print("Top 5 well connected destination nodes:")
# for i in dst_count.index[:6]:
#     print("Node =", node_num[i], "no.of channels =", dst_count[i]) 
    
# print("5 fairly connected destination nodes:")
# for i in dst_count.index[53:58]:
#     print("Node =", node_num[i], "no.of channels =", dst_count[i])


# print("5 poorly connected destination nodes:")
# for i in dst_count.index[-5:]:
#     print("Node =", node_num[i] , "no.of channels =", dst_count[i])
    


node_cap = df[['source', 'satoshis']]
node_cap = node_cap.groupby('source').sum()

well_node = []
fair_node = []
poor_node = []


for i in node_cap.index:
    chan_cnt = src_count[i]
    cap = node_cap.loc[i,'satoshis']
    if cap >= 10**6 and chan_cnt>200:
        well_node.append(node_num[i])
    elif cap > 10**4 and cap < 10**6 and chan_cnt>3 and chan_cnt<=200:
        fair_node.append(node_num[i])
    else:
        poor_node.append(node_num[i])






