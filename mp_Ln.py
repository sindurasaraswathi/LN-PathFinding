# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:08:01 2023

@author: sindu
"""

   
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
# import pickle

startTime = datetime.datetime.now()
 

config = configparser.ConfigParser()
config.read('config.ini')

        
#--------------------------------------------
global use_log, case
epoch = int(config['General']['iterations'])
cbr = int(config['General']['cbr'])
src_type = config['General']['source_type']
dst_type = config['General']['target_type']
amt_type = config['General']['amount_type']
#LND
attemptcost = float(config['LND']['attemptcost'])
attemptcostppm = float(config['LND']['attemptcostppm'])
timepref = float(config['LND']['timepref'])
apriori = float(config['LND']['apriori'])
rf = float(config['LND']['riskfactor'])

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
linear_success_prob = False
min_liq_offset = 0
max_liq_offset = 0
liquidity_penalty_multiplier = 30000/1000
liquidity_penalty_amt_multiplier = 192/1000
hist_liquidity_penalty_multiplier = 10000/1000
hist_liquidity_penalty_amt_multiplier = 64/1000

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
        G.edges[u,v]['capacity'] = int(df['satoshis'][i])
        G.edges[u,v]['Age'] = block_height 
        G.edges[u,v]['BaseFee'] = df['base_fee_millisatoshi'][i]/1000
        G.edges[u,v]['FeeRate'] = df['fee_per_millionth'][i]/1000000
        G.edges[u,v]['Delay'] = df['delay'][i]
        G.edges[u,v]['htlc_min'] = int(re.split(r'(\d+)', df['htlc_minimum_msat'][i])[1])/1000
        G.edges[u,v]['htlc_max'] = int(re.split(r'(\d+)', df['htlc_maximum_msat'][i])[1])/1000
        G.edges[u,v]['LastFailure'] = 25
        x = rn.uniform(0, int(df['satoshis'][i]))
        G.edges[u,v]['Balance'] = x
    return G

      
G = nx.DiGraph()
G = make_graph(G)

def callable(source, target, amt, result, name):
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
                prob_eclair = {} 
                paths = {source:[source]}
                dist = shortest_path_func(G, source=source, 
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
            
                
    #v - target, u - source, d - G.edges[v,u]
    def lnd_cost(v,u,d):
        global timepref
        compute_fee(v,u,d)        
        timepref *= 0.9
        defaultattemptcost = attemptcost+attemptcostppm*amt_dict[(u,v)]/1000000
        penalty = defaultattemptcost * (1/(0.5-timepref/2) - 1)
        prob_weight = 2**G.edges[u,v]["LastFailure"]
        prob = apriori * (1-(1/prob_weight))
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
        if v == target:
            total_prob = prob
        else:
            total_prob = prob * prob_eclair[(v, prev_dict[v][0])]
        if prob<0:
            total_prob = 0
        prob_eclair[(u,v)] = total_prob
        #fee
        total_fee = amt_dict[(u,v)] - amt
        #total CLTV
        total_cltv = G.edges[u,v]["Delay"]
        temp_path = paths[v]
        for i in range(1,len(temp_path)-1):
            p = temp_path[i+1]
            q = temp_path[i]
            total_cltv += G.edges[(p,q)]["Delay"]
        #total Amount
        total_amount = amt_dict[(u,v)]
        #risk cost
        risk_cost = total_amount * G.edges[u,v]["Delay"] * locked_funds_risk
        total_risk_cost = total_amount * total_cltv * locked_funds_risk
        #failure cost
        failure_cost = fail_base + total_amount * fail_rate
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
                if total_prob:
                    cost = total_fee + hopcost + total_risk_cost + failure_cost/total_prob
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
        if linear_success_prob:
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
                fee = round(fee, 5)
                if amount > G.edges[u,v]["Balance"] or amount<=0:
                    G.edges[u,v]["LastFailure"] = 0
                    j = i-1
                    release_locked(j, path)
                    return [path, total_fee, total_delay, path_length, 'Failure']
                else:
                    G.edges[u,v]["Balance"] -= amount
                    G.edges[u,v]["Locked"] = amount  
                    G.edges[u,v]["LastFailure"] = 25
                amount = round(amount - fee, 5)
                if v == target and amount!=amt:
                    print(amt_list)
                    return [path, total_fee, total_delay, path_length, 'Failure']
                
            release_locked(i-1, path)
            return [path, total_fee, total_delay, path_length, 'Success']
        except Exception as e:
            print(e)
            return "Routing Failed due to the above error"
    
    #----------------------------------------------
    def helper(name, func):
        global use_log, case
        try:
            print("\n**",name,"**")
            if name != 'Eclair':
                dist = nx2._dijkstra(G, source=target, target=source, weight = func, pred=prev_dict, paths=paths)
                res = paths[source]
                print("Path found by", name, res[::-1])
                result[name] = route(G, res, source, target)
                
            else:
                for cs in ['Eclair_case1', 'Eclair_case2', 'Eclair_case3']:
                    use_log = config[cs]['use_log']
                    case = config[cs]['case'] 
                    res = list(islice(shortest_simple_paths(G, source=target, target=source, weight=func), 1))
                    for path in res:
                        print("Path found by", cs, path[::-1])
                        result[cs] = route(G, path, source, target)
        except Exception as e:
            print(e)
            
    algo = {'LND':lnd_cost, 'CLN':cln_cost, 'LDK':ldk_cost, 'Eclair':eclair_cost} 
    # for name in algo:
    global fee_dict, amt_dict, cache_node, visited
    global prev_dict, paths
    fee_dict = {}
    amt_dict = {}
    cache_node = target
    visited = set()
    prev_dict = {}
    paths = {target:[target]}
    helper(name, algo[name])
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
        if amt <= upper_bound:
            return True
        else:
            return False
        
        
    work = []              
    result_list = [] 
    
    well_node, fair_node, poor_node = node_classifier()
    i = 0
    while i<epoch:
        if amt_type == 'fixed':
            amt = int(config['General']['amount'])
            
        elif amt_type == 'random':
            k = (i%8)+1
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
        
        for algo in ['LND', 'LDK', 'CLN', 'Eclair']:
            work.append((source, target, amt, result, algo))
        i = i+1
        
    # # with open("data1.pickle", 'wb') as f:
    # #     pickle.dump(work, f)
        
    # with open("data1.pickle", 'rb') as f:
    #     work = pickle.load(f)
    
    pool = mp.Pool(processes=4)
    a = pool.starmap(callable, work)
    result_list.append(a)
    
    
    ans_list = []
    i = 0
    temp = {}
    for res in result_list[0]:
        if i<4:
            temp.update(res)
        i = i+1
        if i == 4:
            ans_list.append(temp)
            temp = {}
            i = 0
            

    fields = ['Source', 'Target', 'Amount', 'LND', 'CLN', 'LDK', 'Eclair_case1', 'Eclair_case2', 'Eclair_case3']
    filename = config['General']['filename'] 
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in ans_list:
            writer.writerow(i)
            
endTime = datetime.datetime.now()
print(endTime - startTime)
    
