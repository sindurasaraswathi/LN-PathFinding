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
import networkx.algorithms.shortest_paths.weighted as nx2

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
        G.edges[u,v]['Amount'] = amt
    return G
amt = 1000        
G = nx.DiGraph()
cbr = 815700
G = make_graph(G)
#-----------------------------------------------------------------------------

def sub_func(u,v, amount):
    global amt_dict, fee_dict
    fee = G.edges[u,v]["BaseFee"] + amount*G.edges[u,v]["FeeRate"]
    fee_dict[(u,v)] = fee
    amt_dict[(u,v)] = amount+fee
 
            
def compute_fee(v,u,d):
    # print("here", v,u, prev_dict)
    global fee_dict, amt_dict, visited, cache_node
    if v == target:
        cache_node = v
        sub_func(u,v,amt)
    else:
        if cache_node != v:
            visited.append(cache_node)
            cache_node = v
        amount = amt_dict[(v, prev_dict[v][0])]        
        sub_func(u,v, amount)
        
            
#v - target, u - source, d - G.edges[v,u]
def lnd_cost(v,u,d):
    global timepref
    rf = 15*10**-9
    compute_fee(v,u,d)        
    timepref *= 0.9
    defaultattemptcost = attemptcost+attemptcostppm*amt_dict[(u,v)]/1000000
    penalty = defaultattemptcost * (1/(0.5-timepref/2) - 1)
    prob_weight = 2**d["LastFailure"]
    prob = apriori * (1-1/prob_weight)
    if prob == 0:
        cost = float('inf')
    else:
        cost = fee_dict[(u,v)] + d['Delay']*amt_dict[(u,v)]*rf + penalty/prob
    return cost
        

def cln_cost(v,u,d):
    rf = 10
    compute_fee(v,u,d)
    cost = amt_dict[(u,v)]*(1+(rf*d["Delay"])/(blk_per_year*100))+1
    return cost


def normalize(value, minm, maxm):
    norm = 0.00001 + 0.99998 * (min(max(minm,value), maxm))/(maxm - minm)
    return norm


def eclair_cost(v,u,d):
    compute_fee(v,u,d)
    ncap = 1-normalize(d["capacity"], min_cap, max_cap)
    nage = normalize(d["Age"], d["Age"]-365*24*6, cbr)
    ncltv = normalize(d["Delay"], min_cltv, max_cltv)
    cost = (fee_dict[(u,v)]+hopcost)*(basefactor + (ncltv*cltvfactor)+
                          (nage*agefactor)+(ncap*capfactor))
    return cost


def ldk_cost(v,u,d):
    htlc_minimum = d['htlc_min']
    compute_fee(v,u,d)
    penalty = 500 + (8192*amt_dict[(u,v)])/2**30
    cost = max(fee_dict[(u,v)], htlc_minimum) + penalty
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
        amt_list = []
        total_fee = 0
        for i in range(len(path)-1):
            v = path[i]
            u = path[i+1]
            if v == target:
                amt_list.append(amt)
            amt_list.append(amt_dict[(u,v)])
            total_fee +=  fee_dict[(u,v)] 
        print("Amount list", amt_list[::-1])
        print("Total fee is ", total_fee)
        path = path[::-1]
        amt_list = amt_list[::-1]
        amount = amt_list[0]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            fee = G.edges[u,v]["BaseFee"] + amt_list[i+1]*G.edges[u,v]["FeeRate"]
            if amount > G.edges[u,v]["Balance"] or amount<=0:
                G.edges[u,v]["LastFailure"] = 0
                j = i-1
                release_locked(j, path)
                return f"Routing failed due to low balance in edge {u},{v}"
            else:
                G.edges[u,v]["Balance"] -= amount
                G.edges[u,v]["Locked"] = amount  
                G.edges[u,v]["LastFailure"] = 25
            amount = amount - fee
            if v == target and amount!=amt:
                print("Amount is", amount)
                return "Routing Failed"
            
        release_locked(i-1, path)
        return f"Routing Successful with total fee = {total_fee}"
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

#----------------------------------------------
def helper(name, func):
    try:
        print("**",name,"**")
        if name != 'Eclair':
            dist = nx2._dijkstra(G, source=target, target=source, weight = func, pred=prev_dict, paths=paths)
            res = paths[source]
            print("Path found by", name, res[::-1])
            print(route(G, res, source, target))
        else:
            res = list(islice(nx.shortest_simple_paths(G, target, source, func), 5))
            for path in res:
                print(path[::-1])
                print(route(G, path, source, target))
    except Exception as e:
        print(e)
        
algo = {'LND':lnd_cost, 'CLN':cln_cost, 'LDK':ldk_cost, 'Eclair': eclair_cost}      
# algo = {'LND':lnd_cost}
# source = 5
# target = 1995
for i in range(1): 
    source = -1
    target = -1
    while (target == source or (source not in G.nodes()) or (target not in G.nodes())):
        target = rn.randint(0, 13129)
        source = rn.randint(0, 13129)
    print("\nSource = ",source, "Target = ", target)
    print("----------------------------------------------")
    for name in algo:
        global fee_dict, amt_dict, visited, cache_node
        global prev_dict, paths
        fee_dict = {}
        amt_dict = {}
        visited = []
        cache_node = target
        prev_dict = {}
        paths = {target:[target]}
        helper(name, algo[name])
    