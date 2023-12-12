# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:08:01 2023

@author: sindu
"""

import networkx as nx
import random as rn
from itertools import islice
import pandas as pd
import re

def make_graph(G):
    df = pd.read_csv('LN_snapshot.csv')
    is_multi = df["short_channel_id"].value_counts() > 1
    df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]
    node_num = {}
    nodes_pubkey = list(set(list(df['source']) + list(df['destination'])))
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
        G.edges[u,v]['htlc_min'] = int(re.split(r'(\d+)', df['htlc_minimum_msat'][i])[1])
        G.edges[u,v]['htlc_max'] = int(re.split(r'(\d+)', df['htlc_maximum_msat'][i])[1])
        G.edges[u,v]['LastFailure'] = 25
        x = rn.uniform(0, int(df['satoshis'][i]))
        G.edges[u,v]['Balance'] = x
        G.edges[u,v]['Amount'] = float('inf')
    return G
        
G = nx.DiGraph()
cbr = 815700
G = make_graph(G)

#-----------------------------------------------------------------------------
def compute_fee(v,u,d,name):
    if v == target or v not in prev_amt:
        prev_amt[v] = amt
        amount = amt
    else:
        amount = prev_amt[v]
    if G.edges[u,v]["FeeRate"] < 1:
        curr_amt = (amount + G.edges[u,v]["BaseFee"])/(1-G.edges[u,v]["FeeRate"])   
    else:
        return float('inf')
    fee = G.edges[u,v]["BaseFee"] + curr_amt*G.edges[u,v]["FeeRate"]
    if u in prev_amt:
        prev_amt[u] = min(curr_amt, prev_amt[u])
    else:
        prev_amt[u] = curr_amt
    G.edges[u,v]["Amount"] = prev_amt[u]
    return fee

#v - target, u - source, d - G.edges[v,u]
def lnd_cost(v,u,d):
    global timepref
    rf = 15*10**-9
    fee = compute_fee(v,u,d,'LND')        
    timepref *= 0.9
    defaultattemptcost = attemptcost+attemptcostppm*G.edges[u,v]['Amount']/1000000
    penalty = defaultattemptcost * (1/(0.5-timepref/2) - 1)
    prob_weight = 2**d["LastFailure"]
    prob = apriori * (1-1/prob_weight)
    if prob == 0:
        cost = float('inf')
    else:
        cost = fee + d['Delay']*G.edges[u,v]['Amount']*rf + penalty/prob
    return cost


def cln_cost(v,u,d):
    rf = 10
    fee = compute_fee(v,u,d, 'CLN')
    cost = G.edges[u,v]['Amount']*(1+(rf*d["Delay"])/(blk_per_year*100))+1
    return cost


def normalize(value, minm, maxm):
    norm = 0.00001 + 0.99998 * (min(max(minm,value), maxm))/(maxm - minm)
    return norm


def eclair_cost(v,u,d):
    fee = compute_fee(v,u,d, 'Eclair')
    ncap = 1-normalize(d["capacity"], min_cap, max_cap)
    nage = normalize(d["Age"], d["Age"]-365*24*6, cbr)
    ncltv = normalize(d["Delay"], min_cltv, max_cltv)
    cost = (fee+hopcost)*(basefactor + (ncltv*cltvfactor)+
                          (nage*agefactor)+(ncap*capfactor))
    return cost


def ldk_cost(v,u,d):
    htlc_minimum = d['htlc_min']
    fee = compute_fee(v,u,d, 'LDK')
    penalty = 500 + (8192*G.edges[u,v]['Amount'])/2**30
    cost = max(fee, htlc_minimum) + penalty
    return cost


def release_locked(j, path):
    while j>=0:
        u = path[j]
        v = path[j+1]
        G.edges[u,v]["Balance"] += G.edges[u,v]["Locked"]
        G.edges[u,v]["Locked"] = 0
        j -= 1
       

def route(G, path, source, target):
    try:
        path = path[::-1]
        amount = G.edges[source, path[1]]['Amount']
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            fee = G.edges[u,v]["BaseFee"] + amount*G.edges[u,v]["FeeRate"]
            # if v == target:
            #     fee = 0  
            if amount > G.edges[u,v]["Balance"] or amount<=0:
                G.edges[u,v]["LastFailure"] = 0
                j = i-1
                release_locked(j, path)
                return "Routing failed"
            else:
                G.edges[u,v]["Balance"] -= amount
                G.edges[u,v]["Locked"] = amount  
                G.edges[u,v]["LastFailure"] = 25
            amount = amount - fee
            if v == target and amount!=amt:
                return "Routing Failed"
            
        release_locked(i-1, path)
        return "Routing Successful"
    except Exception as e:
        print(e)
        return "Routing Failed due to the above error"
        
#--------------------------------------------
amt = 1000
attemptcost = 100
attemptcostppm = 1000
timepref = 0
apriori = 0.6
max_distance_cln = 20
blk_per_year = 52596
cln_bias = 1
max_cap = 10**8
min_cap = 1
min_cltv = 9
max_cltv = 2016
agefactor = 0.35
basefactor = 0
capfactor = 0.5
cltvfactor = 0.15
hopcost = 0 #relay fees
# source = 5
# target = 3
#----------------------------------------------
def helper(name, func):
    try:
        print("**",name,"**")
        if name != 'Eclair':
            res = nx.dijkstra_path(G, target, source, func)
            print(res)
            print(route(G, res, source, target))
        else:
            res = list(islice(nx.shortest_simple_paths(G, target, source, func), 2))
            print(res)
            for path in res:
                print(route(G, path, source, target))
    except Exception as e:
        print(e)
        
algo = {'LND':lnd_cost, 'CLN':cln_cost, 'LDK':ldk_cost, 'Eclair': eclair_cost}      
# algo = {'LND':lnd_cost}
for i in range(1): 
    source = -1
    target = -1
    while (target == source or (source not in G.nodes()) or (target not in G.nodes())):
        target = rn.randint(0, 13129)
        source = rn.randint(0, 13129)
    print("\nSource = ",source, "Target = ", target)
    print("----------------------------------------------")
    for name in algo:
        prev_amt = {}
        helper(name, algo[name])
    