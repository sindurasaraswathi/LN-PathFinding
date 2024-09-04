# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:08:01 2023

@author: sindu
"""
#new remove
#uncomment - do it
   
import datetime
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
import configparser
import csv
from ordered_set import OrderedSet
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# import pickle

startTime = datetime.datetime.now()
 

config = configparser.ConfigParser()
config.read('config.ini')
      
#--------------------------------------------
global use_log, case
global prob_check, prob_dict
prob_check = {}
prob_dict = {}
epoch = int(config['General']['iterations'])
cbr = int(config['General']['cbr'])
src_type = config['General']['source_type']
dst_type = config['General']['target_type']
amt_type = config['General']['amount_type']
#LND
attemptcost = int(config['LND']['attemptcost'])/1000
attemptcostppm = int(config['LND']['attemptcostppm'])
timepref = float(config['LND']['timepref'])
apriori = float(config['LND']['apriori'])
rf = float(config['LND']['riskfactor'])
capfraction = float(config['LND']['capfraction'])
smearing = float(config['LND']['smearing'])


#CLN
max_distance_cln = int(config['CLN']['max_distance_cln'])
blk_per_year = int(config['CLN']['blk_per_year'])
cln_bias = int(config['CLN']['cln_bias'])
rf_cln = int(config['CLN']['riskfactor_cln'])
#Eclair
max_cap = int(config['Eclair']['max_cap'])
min_cap = int(config['Eclair']['min_cap'])
min_cltv = float(config['Eclair']['min_cltv'])
max_cltv = float(config['Eclair']['max_cltv'])
agefactor = float(config['Eclair']['agefactor'])
basefactor = float(config['Eclair']['basefactor'])
capfactor = float(config['Eclair']['capfactor'])
cltvfactor = float(config['Eclair']['cltvfactor'])
hop_base = int(config['Eclair']['hop_base'])/1000
hop_rate = int(config['Eclair']['hop_rate'])/1000000
fail_base = int(config['Eclair']['fail_base'])/1000
fail_rate = int(config['Eclair']['fail_rate'])/1000000
locked_funds_risk = float(config['Eclair']['locked_funds_risk'])
#LDK
base_penalty = float(config['LDK']['base_penalty'])
multiplier = float(config['LDK']['multiplier'])
linear_success_prob = config['LDK']['linear_prob']
min_liq_offset = float(config['LDK']['min_liq_offset'])
max_liq_offset = float(config['LDK']['max_liq_offset'])
liquidity_penalty_multiplier = float(config['LDK']['l_pen_mul'])/1000
liquidity_penalty_amt_multiplier = float(config['LDK']['l_pen_amt_mul'])/1000
hist_liquidity_penalty_multiplier = float(config['LDK']['h_pen_mul'])/1000
hist_liquidity_penalty_amt_multiplier = float(config['LDK']['h_pen_amt_mul'])/1000


#---------------------------------------------------------------------------
def make_graph(G):
    df = pd.read_csv('LN_snapshot.csv')
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
        G.edges[u,v]['capacity'] = int(df['satoshis'][i])#uncomment
        # G.edges[u,v]['capacity'] = 10**8 #new
        G.edges[u,v]['UpperBound'] = int(df['satoshis'][i])
        # G.edges[u,v]['UpperBound'] = 10**8 #new
        G.edges[u,v]['LowerBound'] = 0
        G.edges[u,v]['Age'] = block_height 
        G.edges[u,v]['BaseFee'] = df['base_fee_millisatoshi'][i]/1000
        G.edges[u,v]['FeeRate'] = df['fee_per_millionth'][i]/1000000
        G.edges[u,v]['Delay'] = df['delay'][i]
        G.edges[u,v]['htlc_min'] = int(re.split(r'(\d+)', df['htlc_minimum_msat'][i])[1])/1000
        G.edges[u,v]['htlc_max'] = int(re.split(r'(\d+)', df['htlc_maximum_msat'][i])[1])/1000
        G.edges[u,v]['LastFailure'] = 25
    return G

      
G = nx.DiGraph()
G = make_graph(G)

y = []
cc = 0
#Sample balance from bimodal or uniform distribution
for i in G.edges:#new
    if 'Balance' not in G.edges[i]:
        cap = G.edges[i]['capacity']
        datasample = config['General']['datasampling']
        if datasample == 'bimodal':
            rng = np.linspace(0, cap, 10000)
            s = cap/10
            P = np.exp(-rng/s) + np.exp((rng - cap)/s)
            P /= np.sum(P)            
            x = int(np.random.choice(rng, p=P))
            # if cc<5:
            #     plt.plot(P)
            #     plt.show()
            #     cc += 1
        else:
            x = int(rn.uniform(0, G.edges[i]['capacity']))
            
        (u,v) = i
        G.edges[(u,v)]['Balance'] = x
        G.edges[(v,u)]['Balance'] = cap - x
        
        y.append(x)
        y.append(cap-x)
        
        if G.edges[v,u]['Balance'] < 0 or G.edges[v,u]['Balance'] > G.edges[i]['capacity']:
            print(i, 'Balance error at', (v,u))
            raise ValueError
            
        if G.edges[u,v]['Balance'] < 0 or G.edges[u,v]['Balance'] > G.edges[i]['capacity']:
            print(i, 'Balance error at', (u,v))
            raise ValueError
            
        if G.edges[(v,u)]['Balance'] + G.edges[(u,v)]['Balance'] != cap:
            print('Balance error at', (v,u))
            raise ValueError

plt.hist(y)
plt.show()


def callable(source, target, amt, result, name):
    def tracker(path, dist, p_amt, p_dist):
        global amt_dict
        amt_tracker = {}
        dist_tracker = {}
        # prob_tracker = {}
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
        dist_tracker[u] = dist[u]
        return amt_tracker, dist_tracker
    
    
    def shortest_simple_paths(G, source, target, weight):
        global prev_dict, paths, amt_dict, fee_dict, visited
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
        # prob_holder = PathBuffer()
        prev_path = None
        prev_dist = None
        prev_amt = None
        # prev_prob = None
        visited = set()
        while True:
            if not prev_path:
                prev_dict = {}
                # prob_eclair = {} 
                paths = {source:[source]}
                dist = shortest_path_func(G, source=source, 
                                          target=target, 
                                          weight=weight, 
                                          pred=prev_dict, 
                                          paths=paths)
                path = paths[target]
                visited = set()
                amt_tracker, dist_tracker = tracker(path, dist, prev_amt, prev_dist)
                length = dist_tracker[target]
                listB.push(length, path)
                amt_holder.push(length, amt_tracker)
                dist_holder.push(length, dist_tracker)
                # prob_holder.push(length, prob_tracker)
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
                    # prob_eclair = {}
                    if root[-1] != source:
                        temp_amt = prev_amt[(root[-1], root[-2])]
                        amt_dict[root[-1], root[-2]] = temp_amt
                        prev_dict = {root[-1]:[root[-2]]}
                        # prob_eclair[(root[-1], root[-2])] = prev_prob[(root[-1], root[-2])]
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
                            amt_tracker, dist_tracker = tracker(path, dist, prev_amt, prev_dist)#
                            length = dist[target]
                            listB.push(root_length + length, path)
                            amt_holder.push(root_length + length, amt_tracker)
                            dist_holder.push(root_length + length, dist_tracker)
                            # prob_holder.push(root_length + length, prob_tracker)
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
                # prev_prob = prob_holder.pop()
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
        
    
    #-----------------------------------------------------------------------------
    
    def sub_func(u,v, amount):
        global amt_dict, fee_dict
        fee = G.edges[u,v]["BaseFee"] + amount*G.edges[u,v]["FeeRate"]
        fee_dict[(u,v)] = fee
        if u==source:
            fee_dict[(u,v)] = 0
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
    
    
    def primitive(c, x):
        s = 3e8 #fine tune 's' for improved performance
        ecs = math.exp(-c/s)
        exs = math.exp(-x/s)
        excs = math.exp((x-c)/s)
        norm = -2*ecs + 2
        return (excs - exs)/norm
    
    
    def integral(cap, lower, upper):
        return primitive(cap, upper) - primitive(cap, lower)
    
    
    def bimodal(cap, a_f, a_s, a):
        prob = integral(cap, a, a_f)
        if prob is math.nan:
            return 0
        reNorm = integral(cap, a_s, a_f)
        
        if reNorm is math.nan or reNorm == 0:
            return 0
        prob /= reNorm
        if prob>1:
            return 1
        if prob<0:
            return 0
        return prob
    
    
    #v - target, u - source, d - G.edges[v,u]
    def lnd_cost(v,u,d):
        # global prob_check, prob_dict#new
        global timepref, case
        compute_fee(v,u,d)        
        timepref *= 0.9
        defaultattemptcost = attemptcost+attemptcostppm*amt_dict[(u,v)]/1000000
        penalty = defaultattemptcost * (1/(0.5-timepref/2) - 1)
        cap = G.edges[u,v]["capacity"]
        if case == 'apriori':
            prob_weight = 2**G.edges[u,v]["LastFailure"]
            den = 1+math.exp(-(amt_dict[(u,v)] - capfraction*cap)/(smearing*cap))
            nodeprob = apriori * (1-(0.5/den))
            prob = nodeprob * (1-(1/prob_weight))
            # prob_check[u,v] = prob
        elif case == 'bimodal':
            prob = bimodal(cap, G.edges[u,v]['UpperBound'], G.edges[u,v]["LowerBound"], amt_dict[(u,v)]) 
            # prob_dict[(v,u)] = prob
        if prob == 0:
            cost = float('inf')
        else:
            cost = fee_dict[(u,v)] + G.edges[u,v]['Delay']*amt_dict[(u,v)]*rf + penalty/prob
        return cost

    
    def cln_cost(v,u,d):
        compute_fee(v,u,d)
        cap = G.edges[u,v]['capacity']
        fee = fee_dict[(u,v)]
        curr_amt = amt_dict[(u,v)] - fee
        try:
            cap_bias = math.log(cap+1) - math.log(cap+1-curr_amt)
        except:
            cap_bias = float('inf')
        cost = (fee+((curr_amt*rf_cln*G.edges[u,v]["Delay"])/(blk_per_year*100))+1)*(cap_bias+1)
        return cost
    
    
    def normalize(value, minm, maxm):
        norm = 0.00001 + 0.99998 * (min(max(minm,value), maxm) - minm)/(maxm - minm)
        return norm
    
    
    def eclair_cost(v,u,d):
        global visited
        if u in visited:
            return float('inf')
        compute_fee(v,u,d)
        ncap = 1-normalize(G.edges[u,v]["capacity"], min_cap, max_cap)
        nage = normalize(G.edges[u,v]["Age"], G.edges[u,v]["Age"]-365*24*6, cbr)
        ncltv = normalize(G.edges[u,v]["Delay"], min_cltv, max_cltv)
        
        if v == target:
            hop_amt = amt
        else:
            hop_amt = amt_dict[(v, prev_dict[v][0])]
        hopcost =  hop_base + hop_amt * hop_rate
        #Success Probability
        if G.edges[u,v]["capacity"] != 0:
            prob = 1 - (hop_amt/G.edges[u,v]["capacity"])
        else:
            prob = 0        
        #risk cost
        risk_cost = amt_dict[(u,v)] * G.edges[u,v]["Delay"] * locked_funds_risk
        #failure cost
        failure_cost = fail_base + amt_dict[(u,v)] * fail_rate
        if case == 'WeightRatios':
            cost = (fee_dict[(u,v)]+hopcost)*(basefactor + (ncltv*cltvfactor)+
                              (nage*agefactor)+(ncap*capfactor))
        else:
            if use_log == "True":
                if prob>0:
                    cost = fee_dict[(u,v)] + hopcost + risk_cost - failure_cost * math.log(prob)
                else:
                    cost = float('inf')
            else: 
                if prob>0:
                    cost = fee_dict[(u,v)] + hopcost + risk_cost + failure_cost/prob
                else:
                    cost = float('inf')
        return cost
    
    
    def ldk_neg_log10(num, den):
        return 2048*(math.log10(den) - math.log10(num))
    
    
    def ldk_combined_penalty(a, neg_log, liquidity_penalty_mul, liquidity_penalty_amt_mul):
        neg_log = min(neg_log, 2*2048)
        liq_penalty = neg_log * liquidity_penalty_mul/2048
        amt_penalty = neg_log * liquidity_penalty_amt_mul * a/(2048 * 2**20)
        return liq_penalty + amt_penalty
    
    
    def ldk_prob(a, min_liq, max_liq, cap, success_flag):
        min_liquidity = min_liq
        if linear_success_prob == 'True':
            num = max_liq - a
            den = max_liq - min_liq + 1
        else:
            min_liq = min_liq/cap
            max_liq = max_liq/cap
            a = a/cap
            num = ((max_liq-0.5)**3) - ((a-0.5)**3)
            den = ((max_liq-0.5)**3) - ((min_liq-0.5)**3)
            billionish = 1024**3
            num = (num*billionish) + 1
            den = (den*billionish) + 1
        if ((success_flag and min_liquidity) == 0) and den<(((2**64)-1)/21):
            den = den*21/16
        return num, den
            
    
    def liq_penalty(v,u):
        capacity = G.edges[u,v]["capacity"]
        max_liquidity = capacity - max_liq_offset
        min_liquidity = min(min_liq_offset, max_liquidity)
        a = amt_dict[(u,v)]
        if  a <= min_liquidity:
            res = 0
        elif a >= max_liquidity:
            res = ldk_combined_penalty(a, 2*2048, liquidity_penalty_multiplier, liquidity_penalty_amt_multiplier)
        else:
            (num, den) = ldk_prob(a, min_liquidity, max_liquidity, capacity, False)
            if (den-num)<(den/64):
                res = 0
            else:
                neg_log = ldk_neg_log10(num, den)
                res = ldk_combined_penalty(a, neg_log, liquidity_penalty_multiplier, liquidity_penalty_amt_multiplier)
        if a >= capacity:
            res = res + ldk_combined_penalty(a, 2*2048, hist_liquidity_penalty_multiplier, hist_liquidity_penalty_amt_multiplier)
            return res
        if hist_liquidity_penalty_multiplier != 0 or hist_liquidity_penalty_amt_multiplier!=0:
            (num, den) = ldk_prob(a, 0, capacity, capacity, True)
            neg_log = ldk_neg_log10(num, den)
            res = res + ldk_combined_penalty(a, neg_log, hist_liquidity_penalty_multiplier, hist_liquidity_penalty_amt_multiplier)
        return res
    
    def final_penalty(v,u):
        htlc_max = G.edges[u,v]["htlc_max"]
        anti_probing_penalty = 0
        if htlc_max >= G.edges[u,v]["capacity"]/2:
            anti_probing_penalty = 250/1000
        penalty_base = base_penalty/1000 + ((multiplier/1000)*amt_dict[(u,v)])/2**30
        penalty_liquidity = liq_penalty(v,u)
        penalty_total = penalty_base + penalty_liquidity + anti_probing_penalty
        return penalty_total
            

    def ldk_cost(v,u,d):
        htlc_minimum = G.edges[u,v]['htlc_min']
        # curr_min = max(nextHopHtlcmin, htlc_minimum)
        htlc_fee = htlc_minimum * G.edges[u,v]['FeeRate'] + G.edges[u,v]['BaseFee']
        path_htlc_minimum = htlc_fee + htlc_minimum
        compute_fee(v,u,d)
        penalty = final_penalty(v,u)
        cost = max(fee_dict[(u,v)], path_htlc_minimum) + penalty
        return cost
    
    
    def release_locked(j, path):
        while j>=0:
            u = path[j]
            v = path[j+1]
            G.edges[u,v]["Balance"] += G.edges[u,v]["Locked"]
            G.edges[u,v]["Locked"] = 0
            j -= 1
           
    
    #simulate payment routing on the path found by the LN clients
    def route(G, path, source, target):
        try:
            amt_list = []
            total_fee = 0
            total_delay = 0
            path_length = len(path)
            for i in range(path_length-1):
                v = path[i]
                u = path[i+1]
                if v == target:
                    amt_list.append(amt)
                fee = G.edges[u,v]["BaseFee"] + amt_list[-1]*G.edges[u,v]["FeeRate"]
                if u==source:
                    fee = 0
                fee = round(fee, 5)
                a = round(amt_list[-1] + fee, 5)
                amt_list.append(a)
                total_fee +=  fee
                total_delay += G.edges[u,v]["Delay"]
            path = path[::-1]
            amt_list = amt_list[::-1]
            amount = amt_list[0]
            for i in range(path_length-1):
                u = path[i]
                v = path[i+1]
                fee = G.edges[u,v]["BaseFee"] + amt_list[i+1]*G.edges[u,v]["FeeRate"]
                if u==source:
                    fee = 0
                fee = round(fee, 5)
                if amount > G.edges[u,v]["Balance"] or amount<=0:
                    G.edges[u,v]["LastFailure"] = 0
                    if amount < G.edges[u,v]["UpperBound"]:
                        G.edges[u,v]["UpperBound"] = amount #new
                    j = i-1
                    release_locked(j, path)
                    return [path, total_fee, total_delay, path_length, 'Failure']
                else:
                    G.edges[u,v]["Balance"] -= amount
                    G.edges[u,v]["Locked"] = amount  
                    G.edges[u,v]["LastFailure"] = 100
                    if G.edges[u,v]["LowerBound"] < amount:
                        G.edges[u,v]["LowerBound"] = amount #new
                amount = round(amount - fee, 5)
                if v == target and amount!=amt:
                    return [path, total_fee, total_delay, path_length, 'Failure']
          
            release_locked(i-1, path)
            return [path, total_fee, total_delay, path_length, 'Success']
        except Exception as e:
            print(e)
            return "Routing Failed due to the above error"
    
    
    #----------------------------------------------
    def dijkstra_caller(res_name, func):
        dist = nx2._dijkstra(G, source=target, target=source, weight = func, pred=prev_dict, paths=paths)
        res = paths[source]
        print("Path found by", res_name, res[::-1])
        result[res_name] = route(G, res, source, target)
        
        
    def helper(name, func):
        global use_log, case
        try:
            print("\n**",name,"**")
            if name != 'Eclair':
                if name == 'LND':
                    lndcase = config['General']['lndcase'].split('|')
                    for cs in lndcase:
                        case = config[name][cs]
                        dijkstra_caller(cs, func)
                        prob_dict[cs] = prob_check
                else:
                    dijkstra_caller(name, func)
            else:
                eclaircase = config['General']['eclaircase'].split('|')
                for cs in eclaircase:
                    use_log = config[cs]['use_log']
                    case = config[cs]['case'] 
                    res = list(islice(shortest_simple_paths(G, source=target, target=source, weight=func), 1))
                    for path in res:
                        print("Path found by", cs, path[::-1])
                        result[cs] = route(G, path, source, target)
        except Exception as e:
            print(e)
            
    algo = {'LND':lnd_cost, 'CLN':cln_cost, 'LDK':ldk_cost, 'Eclair':eclair_cost} 
    global fee_dict, amt_dict, cache_node, visited
    global prev_dict, paths
    fee_dict = {}
    amt_dict = {}
    cache_node = target
    visited = set()
    prev_dict = {}
    paths = {target:[target]}
    helper(name, algo[name])
    # print(prob_dict)
    # print(prob_check)
    return result
        

if __name__ == '__main__':       
    def node_classifier():
        df = pd.read_csv('LN_snapshot.csv')
        is_multi = df["short_channel_id"].value_counts() > 1
        df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]
        nodes_pubkey = list(OrderedSet(list(df['source']) + list(df['destination'])))
        node_num = {}
        for i in range(len(nodes_pubkey)):
            pubkey = nodes_pubkey[i]
            node_num[pubkey] = i   
        src_count = df['source'].value_counts()
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
        return well_node, fair_node, poor_node
    
    
    def node_selector(node_type):
        if node_type == 'well':
            return rn.choice(well_node)
        elif node_type == 'fair':
            return rn.choice(fair_node)
        elif node_type == 'poor':
            return rn.choice(poor_node)
        else:
            return rn.randint(0,13129)
        
        
    def node_ok(source, target):
        src_max = 0
        tgt_max = 0
        for edges in G.out_edges(source):
            src_max = max(src_max, G.edges[edges]['Balance'])
        for edges in G.in_edges(target):
            tgt_max = max(tgt_max, G.edges[edges]['Balance'])
        upper_bound = int(min(src_max, tgt_max))
        if amt < upper_bound:
            return True
        else:
            return False
        
            
    def node_cap(source, target):
        src_max = 0
        tgt_max = 0
        for edges in G.out_edges(source):
            src_max = max(src_max, G.edges[edges]['Balance'])
        for edges in G.in_edges(target):
            tgt_max = max(tgt_max, G.edges[edges]['Balance'])
        upper_bound = int(min(src_max, tgt_max))
        return upper_bound
        
    work = []              
    result_list = [] 
    prob_dict = {}
    
    well_node, fair_node, poor_node = node_classifier()
    i = 0
    
    algos = config['General']['algos'].split('|')
    amt_end_range = int(config['General']['amt_end_range'])
    #uniform random amount selection
    while i<epoch:
        if amt_type == 'fixed':
            amt = int(config['General']['amount'])
            
        elif amt_type == 'random':
            k = (i%amt_end_range)+1 #i%6 for fair node else i%8 
            amt = rn.randint(10**(k-1), 10**k)
            
        result = {}
        source = -1
        target = -1
        while (target == source or (source not in G.nodes()) or (target not in G.nodes())):
            source = node_selector(src_type)
            target = node_selector(dst_type)
        
        if not(node_ok(source, target)):
            continue 
                     
        print("\nSource = ",source, "Target = ", target, "Amount=", amt, 'Epoch =', i)
        print("----------------------------------------------")
        result['Source'] = source
        result['Target'] = target
        result['Amount'] = amt
        # result['upper bound'] = cap
        
        for algo in algos:#uncomment
            work.append((source, target, amt, result, algo))
        # work.append((source, target, amt, result, 'LND')) #new
        i = i+1
    
    pool = mp.Pool(processes=8)
    a = pool.starmap(callable, work)
    result_list.append(a)
    
    ##4 work so i=4
    ans_list = [] #uncomment
    i = 0
    temp = {}
    for res in result_list[0]:
        if i<len(algos):
            temp.update(res)
        i = i+1
        if i == len(algos):
            ans_list.append(temp)
            temp = {}
            i = 0

            
    # fields = ['Source', 'Target', 'Amount', 'LND1', 'LND2', 'CLN', 'LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3']#uncomment
    fields = ['Source', 'Target', 'Amount'] 
    lndcase = config['General']['lndcase'].split('|')
    eclaircase = config['General']['eclaircase'].split('|')
    if 'LND' in algos:
        if 'LND1' in lndcase:
            fields.append('LND1')
        if 'LND2' in lndcase:
            fields.append('LND2')
    if 'CLN' in algos:
        fields.append('CLN')
    if 'LDK' in algos:
        fields.append('LDK')
    if 'Eclair' in algos:
        if 'Eclair_case1' in eclaircase:
            fields.append('Eclair_case1')
        if 'Eclair_case2' in eclaircase:
            fields.append('Eclair_case2')
        if 'Eclair_case3' in eclaircase:
            fields.append('Eclair_case3')
    
    filename = config['General']['filename'] 
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in ans_list:
            writer.writerow(i)
        
            
endTime = datetime.datetime.now()
print(endTime - startTime)
    