#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:10:22 2024

@author: ssarasw2
"""
import pandas as pd
from statistics import mean, mode, median
from ordered_set import OrderedSet

df = pd.read_csv('C:/Users/sindu/Work/UNCC Research/GIT_LN/LN-PathFinding/LN_snapshot.csv')
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
    
import networkx as nx
import re
import random as rn
def make_graph(G):
    df = pd.read_csv('C:/Users/sindu/Work/UNCC Research/GIT_LN/LN-PathFinding/LN_snapshot.csv')
    is_multi = df["short_channel_id"].value_counts() > 1
    df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]
    node_num = {}
    nodes_pubkey = list(OrderedSet(list(df['source']) + list(df['destination'])))
    for i in range(len(nodes_pubkey)):
        G.add_node(i)
        pubkey = nodes_pubkey[i]
        G.nodes[i]['pubkey'] = pubkey
        node_num[pubkey] = i
    for i in df.index:
        node_src = df['source'][i]
        node_dest = df['destination'][i]
        u = node_num[node_src]
        v = node_num[node_dest]
        G.add_edge(u,v)
        channel_id = df['short_channel_id'][i]
        block_height = int(channel_id.split('x')[0])
        G.edges[u,v]['id'] = channel_id
        G.edges[u,v]['capacity'] = int(df['satoshis'][i])
        G.edges[u,v]['Age'] = block_height 
        G.edges[u,v]['BaseFee'] = df['base_fee_millisatoshi'][i]/1000
        G.edges[u,v]['FeeRate'] = df['fee_per_millionth'][i]/1000000
        G.edges[u,v]['Delay'] = df['delay'][i]
        G.edges[u,v]['htlc_min'] = int(re.split(r'(\d+)', df['htlc_minimum_msat'][i])[1])/1000
        G.edges[u,v]['htlc_max'] = int(re.split(r'(\d+)', df['htlc_maximum_msat'][i])[1])/1000
        G.edges[u,v]['LastFailure'] = 25
        if (v,u) in G.edges:
            G.edges[u,v]['Balance'] = G.edges[u,v]['capacity'] - G.edges[v,u]['Balance']
        else:
            x = int(rn.uniform(0, int(df['satoshis'][i])))
            G.edges[u,v]['Balance'] = x            
    return G

      
G = nx.DiGraph()
G = make_graph(G)

node_cap = df[['source', 'satoshis']]
node_cap = node_cap.groupby('source').sum()

well_node = []
fair_node = []
poor_node = []


for i in node_cap.index:
    chan_cnt = src_count[i]
    cap = node_cap.loc[i,'satoshis']
    if cap >= 10**6 and chan_cnt>5:
        well_node.append(node_num[i])
    elif cap > 10**4 and cap < 10**6 and chan_cnt>5:
        fair_node.append(node_num[i])
    elif chan_cnt<=5:
        poor_node.append(node_num[i])
       
num_edges_w = []
cap_w = []
for w in well_node:
    num_edges_w.append(len(G.out_edges(w)))
    temp = 0
    for i in G.out_edges(w):
        temp += G.edges[i]['capacity']
    cap_w.append(temp)
    
num_edges_f = []
cap_f = []
for w in fair_node:
    num_edges_f.append(len(G.out_edges(w)))
    temp = 0
    for i in G.out_edges(w):
        temp += G.edges[i]['capacity']
    cap_f.append(temp)

num_edges_p = []
cap_p = []
for w in poor_node:
    num_edges_p.append(len(G.out_edges(w)))  
    temp = 0
    for i in G.out_edges(w):
        temp += G.edges[i]['capacity']
    cap_p.append(temp)


print("Mean of edges in Well", mean(num_edges_w))
print("Mean of edges in fair", mean(num_edges_f))
print("Mean of edges in poor", mean(num_edges_p))

print("Mean of capacity in well:", mean(cap_w))
print("Mean of capacity in fair:", mean(cap_f))
print("Mean of capacity in poor:", mean(cap_p))


