# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import trapz
from scipy.signal import welch 
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal

def time_from_list(t_list, t0=0):
    if t_list[0]>9:
        t_sec = -(60-t_list[2] + 60*(59-t_list[1]) + 3600*(11-t_list[0])) -t0
    else:
        t_sec = t_list[2] + 60*t_list[1] + 3600*t_list[0] -t0
    return t_sec
        
def wake_times(filename : str):   
    with open(filename) as f:
        lines = f.readlines()
    t0 = time_from_list([float(x) for x in lines[0].split()])
    t1 = []
    if len(lines) > 2:       
        for line in lines[1:-1]:
            t1.append(time_from_list([float(x) for x in line.split()], t0=t0))
    tf = time_from_list([float(x) for x in lines[-1].split()], t0=t0)
    return t1, tf
    
def find_valid_segs(t_logic, min_len):
    idx_starts = []
    idx_ends = []
    t_logic_list = t_logic.tolist()
    try:
        idx_start = t_logic_list.index(True)
    except ValueError:
        return idx_starts, idx_ends
    idx_end = np.inf
    end_temp = idx_start+1
    while end_temp < t_logic.size:       
        right_cap = min([end_temp+min_len, t_logic.size])
        if np.all(t_logic[end_temp: right_cap]):
            end_temp += min_len
        else:
            junction_list = t_logic_list[end_temp:right_cap]
            idx_end = junction_list.index(False)+end_temp
            if idx_end - idx_start > min_len:
                idx_starts.append(idx_start)
                idx_ends.append(idx_end)
            last_true = junction_list[::-1].index(False)
            if last_true > 0:               
                idx_start = right_cap-last_true 
                idx_end = np.inf
                end_temp = idx_start+1
            else:
                try:
                    idx_start = t_logic_list[right_cap:].index(True)+right_cap
                    idx_end = np.inf
                    end_temp = idx_start+1
                except ValueError:
                    return idx_starts, idx_ends                
    if (idx_end - idx_start > min_len) and (idx_start<t_logic.size):
        idx_starts.append(idx_start)
        idx_ends.append(idx_end)
        return idx_starts, idx_ends

def time_analysis(rt):
    rr = np.diff(rt)
    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    pnn20 = np.where(np.absolute(np.diff(rr))>0.02)[0].size/(rr.size-1.)
    pnn30 = np.where(np.absolute(np.diff(rr))>0.03)[0].size/(rr.size-1.)
    pnn40 = np.where(np.absolute(np.diff(rr))>0.04)[0].size/(rr.size-1.)
    # pnn50 = np.where(np.absolute(np.diff(rr))>0.05)[0].size/(rr.size-1.)
    return mean_rr, sdnn, pnn20, pnn30, pnn40

def freq_analysis(rt):
    rt = np.array(rt)
    f = interp1d(rt[:-1], np.diff(rt), kind='cubic')
    # sample rate for interpolation
    fs = 4.0
    steps = 1 / fs
    newt = np.arange(rt[0], rt[-2], steps)
    rt_intpl = f(newt)
    ft, Pt = welch(rt_intpl, fs)
    # plt.figure()
    # plt.plot(ft, Pt, 'b.-')
    # plt.plot(newt, rt_intpl, 'y.-')
    cond_vlf = (ft>=0) & (ft<0.04)
    cond_lf = (ft>=0.04) & (ft<0.15)
    cond_hf = (ft>=0.15) & (ft<0.4)
    
    vlf = trapz(Pt[cond_vlf], ft[cond_vlf])
    lf = trapz(Pt[cond_lf], ft[cond_lf])
    hf = trapz(Pt[cond_hf], ft[cond_hf])
    return vlf, lf, hf

def prior_func(end_time, prior_params):
    def prior_probs(t):
        c = 3600.
        p_tot = 1.
        probs = [] # W0, FW, D, BW, S, R 
        hill_func = lambda x, t0, n: (x/t0)**n/(1.+(x/t0)**n)
        # W0
        cutoff_time = prior_params['W0_cutoff']*c
        if t<cutoff_time:
            probs.append(1. - hill_func(t, cutoff_time/2., prior_params['W0_n']))
        else:
            probs.append(0)
        #FW
        # cutoff_time = (end_time- prior_params['FW_cutoff'])*c
        # if t>cutoff_time:
        #     probs.append(hill_func(t-cutoff_time, prior_params['FW_cutoff']*c/2., prior_params['FW_n']))
        # else:
        #     probs.append(0)
        probs.append(0)
        p_tot -= sum(probs[:])
        # D
        cutoff_time = prior_params['D_cutoff']*c
        if t<cutoff_time:
            D_hill = hill_func(t, cutoff_time/2., prior_params['D_n'])*prior_params['D_max']    
        else: 
            D_hill = 0
        # BW
        BW_hill = hill_func(t, prior_params['BW_t0']*c, prior_params['BW_n'])*prior_params['BW_max']
        #D, BW, S, R 
        # prob_un = np.array([D_hill, prior_params['BW_const'], prior_params['SR_const'], prior_params['SR_const']])
        prob_un = np.array([D_hill, BW_hill, prior_params['SR_const'], prior_params['SR_const']])
        prob_nor = prob_un/np.sum(prob_un)*p_tot
        probs.extend(prob_nor.tolist())
        return probs
    return prior_probs

def bayes_wrap(dist_list):
    def bayes_classifier(feature, p_prior):
        p_likelihood = [multivariate_normal.pdf(feature, mean=dist[0], cov=dist[1]) for dist in dist_list]
        p_post = np.array(p_likelihood)*np.array(p_prior)/np.sum(np.array(p_likelihood)*np.array(p_prior))
        return p_post
    return bayes_classifier

def on_switch(stage_binary, t_list, w):
    stage_ave = np.convolve(stage_binary, np.ones(w), 'valid')/float(w)
    t_array = np.array(t_list[w-1:])
    stage_bg = np.mean(stage_ave[(t_array>7200) & (t_array<21600)])
    stage_norm = (stage_ave-stage_bg)/stage_bg
    stage_later = stage_norm[t_array>21600]
    if np.any(stage_later>2):
        return True
    else:
        return False