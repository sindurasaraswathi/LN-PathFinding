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
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function
import math

def tracker(path, dist, p_amt, p_dist, p_prob):
    global amt_dict, prob_eclair
    amt_tracker = {}
    dist_tracker = {}
    prob_tracker = {}
    for i in range(len(path)-1):
        u = path[i+1]
        v = path[i]
        if (u,v) in amt_dict:
            amt_tracker[(u,v)] = amt_dict[(u,v)]
        else:
            amt_tracker[(u,v)] = p_amt[(u,v)]
        if v in dist:
            dist_tracker[v] = dist[v]
        else:
            dist_tracker[v] = p_dist[v]
        if (u,v) in prob_eclair:
            prob_tracker[(u,v)] = prob_eclair[(u,v)]
        else:#
            prob_tracker[(u,v)] = p_prob[(u,v)]

    dist_tracker[u] = dist[u]
    return amt_tracker, dist_tracker, prob_tracker


def shortest_simple_paths(G, source, target, weight):
    global prev_dict, paths, amt_dict, fee_dict, visited, prob_eclair
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    wt = _weight_function(G, weight)

    shortest_path_func = nx2._dijkstra
    
    listA = []
    listB = PathBuffer()
    amt_holder = PathBuffer()
    dist_holder = PathBuffer()
    prob_holder = PathBuffer()
    prev_path = None
    prev_dist = None
    prev_amt = None
    prev_prob = None
    visited = set()
    while True:
        if not prev_path:
            prev_dict = {}
            prob_eclair = {} #
            paths = {source:[source]}
            dist = nx2._dijkstra(G, source=source, 
                                      target=target, 
                                      weight=weight, 
                                      pred=prev_dict, 
                                      paths=paths)
            path = paths[target]
            visited = set()
            amt_tracker, dist_tracker, prob_tracker = tracker(path, dist, prev_amt, prev_dist, prev_prob)#
            length = dist_tracker[target]
            listB.push(length, path)
            amt_holder.push(length, amt_tracker)
            dist_holder.push(length, dist_tracker)
            prob_holder.push(length, prob_tracker)
        else:
            # global root,ignore_edges, H
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                visited = set(root.copy())
                root_length = prev_dist[root[-1]]        
                amt_dict = {}
                fee_dict = {}
                prev_dict = {}
                prob_eclair = {}
                if root[-1] != source:
                    temp_amt = prev_amt[(root[-1], root[-2])]
                    amt_dict[root[-1], root[-2]] = temp_amt
                    prev_dict = {root[-1]:[root[-2]]}
                    prob_eclair[(root[-1], root[-2])] = prev_prob[(root[-1], root[-2])]
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    H = G.copy()
                    H.remove_edges_from(ignore_edges)
                    H.remove_nodes_from(ignore_nodes)
                    paths = {root[-1]:[root[-1]]}
                    dist = shortest_path_func(
                        H,
                        source=root[-1],
                        target=target,
                        weight=weight,
                        pred = prev_dict,
                        paths = paths
                        
                    )
                    try:
                        path = root[:-1] + paths[target]
                        amt_tracker, dist_tracker, prob_tracker = tracker(path, dist, prev_amt, prev_dist, prev_prob)#
                        length = dist[target]
                        listB.push(root_length + length, path)
                        amt_holder.push(root_length + length, amt_tracker)
                        dist_holder.push(root_length + length, dist_tracker)
                        prob_holder.push(root_length + length, prob_tracker)
                    except:
                        pass
                except:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
            prev_amt = amt_holder.pop()
            prev_dist = dist_holder.pop()
            prev_prob = prob_holder.pop()
        else:
            break


class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = []
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path
    
#---------------------------------------------------------------------------
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
    global fee_dict, amt_dict, cache_node, visited
    if v == target:
        cache_node = v
        sub_func(u,v,amt)
    else:
        if cache_node != v:
            visited.add(cache_node)
            cache_node = v          
        amount = amt_dict[(v, prev_dict[v][0])] 
        sub_func(u,v,amount)
        
            
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
    global visited
    if u in visited:
        return float('inf')
    compute_fee(v,u,d)
    ncap = 1-normalize(d["capacity"], min_cap, max_cap)
    nage = normalize(d["Age"], d["Age"]-365*24*6, cbr)
    ncltv = normalize(d["Delay"], min_cltv, max_cltv)
    if v == target:
        hop_amt = amt
    else:
        hop_amt = amt_dict[(v, prev_dict[v][0])]
    hopcost =  hop_base + hop_amt * hop_rate
    #Success Probability
    prob = 1 - (hop_amt/d["capacity"])
    if v == target:
        total_prob = prob
    else:
        total_prob = prob * prob_eclair[(v, prev_dict[v][0])]
    prob_eclair[(u,v)] = total_prob
    #fee
    total_fee = amt_dict[(u,v)] - amt
    #total CLTV
    total_cltv = d["Delay"]
    temp_path = paths[v]
    for i in range(1,len(temp_path)-1):
        p = temp_path[i+1]
        q = temp_path[i]
        total_cltv += G.edges[(p,q)]["Delay"]
    #total Amount
    total_amount = amt_dict[(u,v)]
    #risk cost
    risk_cost = total_amount * d["Delay"] * locked_funds_risk
    total_risk_cost = total_amount * total_cltv * locked_funds_risk
    #failure cost
    failure_cost = fail_base + total_amount * fail_rate
    if case == 'WeightRatios':
        cost = (fee_dict[(u,v)]+hopcost)*(basefactor + (ncltv*cltvfactor)+
                          (nage*agefactor)+(ncap*capfactor))
    else:
        if use_log:
            cost = fee_dict[(u,v)] + hopcost + risk_cost - failure_cost * math.log(prob)
        else: 
            cost = total_fee + hopcost + total_risk_cost + failure_cost/total_prob
    return cost


def ldk_cost(v,u,d):
    htlc_minimum = d['htlc_min']
    compute_fee(v,u,d)
    path_htlc_minimum = max(htlc_minimum+fee_dict[(u,v)], htlc_minimum)
    penalty = base_penalty + (multiplier*amt_dict[(u,v)])/2**30
    cost = max(fee_dict[(u,v)], path_htlc_minimum) + penalty
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
            fee = G.edges[u,v]["BaseFee"] + amt_list[-1]*G.edges[u,v]["FeeRate"]
            amt_list.append(amt_list[-1] + fee)
            total_fee +=  fee
        path = path[::-1]
        amt_list = amt_list[::-1]
        print("Amount list", amt_list)
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
hop_base = 0 #relay fees#
hop_rate = 0
fail_base = 2000
fail_rate = 500
locked_funds_risk = 1e-8
use_log = False 
case = 'WeightRatios'
# case = 'Heuristics'
base_penalty = 500
multiplier = 8192

#----------------------------------------------
def helper(name, func):
    try:
        print("\n**",name,"**")
        if name != 'Eclair':
            dist = nx2._dijkstra(G, source=target, target=source, weight = func, pred=prev_dict, paths=paths)
            res = paths[source]
            print("Path found by", name, res[::-1])
            print(route(G, res, source, target))
        else:
            res = list(islice(shortest_simple_paths(G, source=target, target=source, weight=func), 1))
            for path in res:
                print("Path found by", name, path[::-1])
                print(route(G, path, source, target))
    except Exception as e:
        print(e)
        
algo = {'LND':lnd_cost, 'CLN':cln_cost, 'LDK':ldk_cost, 'Eclair':eclair_cost}      
for i in range(1): 
    source = -1
    target = -1
    while (target == source or (source not in G.nodes()) or (target not in G.nodes())):
        target = rn.randint(0, 13129)
        source = rn.randint(0, 13129)
    print("\nSource = ",source, "Target = ", target)
    print("----------------------------------------------")
    for name in algo:
        global fee_dict, amt_dict, cache_node, visited
        global prev_dict, paths
        fee_dict = {}
        amt_dict = {}
        cache_node = target
        visited = set()
        prev_dict = {}
        paths = {target:[target]}
        helper(name, algo[name])
    