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

def shortest_simple_paths(G, source, target, weight=None):
    global prev_dict, paths, amt_tracker, amt_dict, fee_dict
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    wt = _weight_function(G, weight)

    def length_func(path):
        return sum(
            wt(u, v, G.get_edge_data(u, v)) for (u, v) in zip(path, path[1:])
        )

    shortest_path_func = nx2._dijkstra
    
    listA = []
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            prev_dict = {}
            paths = {source:[source]}
            dist = shortest_path_func(G, source=source, 
                                      target=target, 
                                      weight=weight, 
                                      pred=prev_dict, 
                                      paths=paths)
            path = paths[target]
            print(path)
            amt_tracker = {}
            # for i in range(len(path)-1):
            #     amt_tracker[(path[i+1], path[i])] = amt_dict[(path[i+1],path[i])]
            # print(amt_tracker)
            length = dist[target]
            listB.push(length, path)
        else:
            global root,ignore_edges, H
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                print("Root",root)
                fee_dict = {}
                prev_dict = {}
                if root[-1] != source:
                    # amt_dict[root[-1], root[-2]] = amt_tracker[root[-1], root[-2]]
                    temp_amt = amt_dict[root[-1], root[-2]]
                    amt_dict = {}
                    amt_dict[root[-1], root[-2]] = temp_amt
                    prev_dict = {root[-1]:[root[-2]]}
                    print(amt_dict)
                    print(prev_dict)
                else:
                    amt_dict = {}
                
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    
                    H = G.copy()
                    H.remove_edges_from(ignore_edges)
                    paths = {root[-1]:[root[-1]]}
                    print("here")
                    dist = shortest_path_func(
                        H,
                        source=root[-1],
                        target=target,
                        weight=weight,
                        pred = prev_dict,
                        paths = paths
                        
                    )
                    path = root[:-1] + paths[target]
                    print(path)
                    length = dist[target]
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
            
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
    global fee_dict, amt_dict, cache_node
    if v == target:
        cache_node = v
        sub_func(u,v,amt)
    else:
        if cache_node != v:
            cache_node = v
        amount = amt_dict[(v, prev_dict[v][0])] 
        sub_func(u,v, amount)
        
            
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
        path = path[::-1]
        amt_list = amt_list[::-1]
        print("Amount list", amt_list)
        print("Total fee is ", total_fee)
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
            res = list(islice(shortest_simple_paths(G, source=target, target=source, weight=func), 2))
            for path in res:
                print(path[::-1])
                print(route(G, path, source, target))
    except Exception as e:
        print(e)
             
algo = {'Eclair':eclair_cost}
source = 5
target = 1995
for i in range(1): 
    # source = -1
    # target = -1
    # while (target == source or (source not in G.nodes()) or (target not in G.nodes())):
    #     target = rn.randint(0, 13129)
    #     source = rn.randint(0, 13129)
    print("\nSource = ",source, "Target = ", target)
    print("----------------------------------------------")
    for name in algo:
        global fee_dict, amt_dict, cache_node
        global prev_dict, paths
        fee_dict = {}
        amt_dict = {}
        cache_node = target
        helper(name, algo[name])
    