# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:05:52 2022

@author: lxie2
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage  
# import matplotlib.pyplot as plt

# return the candidate of next peak of each peak and the score of each peak
def find_next_peak(peak_list, dist_range):
    dist_low, dist_hi = dist_range
    peak_dist = np.diff(peak_list)
    peak_len = len(peak_list)
    peak_next = []
    peak_score = np.zeros(peak_len, dtype=int)
    for i in range(peak_len-1):
        can_list = []
        next_dist = peak_dist[i]
        j = 0
        while next_dist < dist_hi:
            if dist_low <= next_dist <= dist_hi:
                can_list.append(i+j+1)
            elif next_dist < dist_low:
                peak_score[i+j+1] = peak_score[i+j+1]+1
            j += 1
            if (i+j) < peak_len-1:
                next_dist = next_dist+peak_dist[i+j]
            else:
                break
        if (j==0 and peak_score[i]==0 and next_dist<=1.5*dist_hi):
            can_list.append(i+1)
            peak_score[i] = peak_score[i]+1
        else:
            peak_score[i] = peak_score[i]+j
        peak_next.append(can_list)    
    peak_next.append([])
    return peak_next, peak_score

# remove peaks that are too low (height less than peak_cut of their close neighbours' heights) 
# a close neighbour is less than dist_low away 
def rm_low_peaks(peak_list, peak_height, rm_para):
    dist_low, peak_cut = rm_para
    peak_dist = np.diff(peak_list)
    temp_list = np.where(peak_dist < dist_low)[0]
    peak_height_left = peak_height[temp_list]
    peak_height_right = peak_height[temp_list+1]
    rm_bool_left = peak_height_left < peak_height_right*peak_cut
    rm_bool_right = peak_height_left*peak_cut > peak_height_right
    rm_list = np.union1d(temp_list[rm_bool_left], temp_list[rm_bool_right]+1)
    if len(rm_list)>0:
        peak_list = np.delete(peak_list, rm_list)
        peak_height = np.delete(peak_height, rm_list)
    return peak_list, peak_height

def find_candidate_peaks(hrdata, can_peak_para):
    win_size, sample_rate, adj_per = can_peak_para
    # if a local maximum is larger than the ajusted rolling mean, it's a
    # candidate peak.
    rol_mean = ndimage.uniform_filter1d(hrdata.astype(float), int(win_size*sample_rate))
    mean_adj = np.percentile(hrdata, adj_per)
    # adjust the rolling mean upward by mean_adj
    rol_mean_adj = rol_mean+mean_adj-np.mean(hrdata)

    peaksx = np.where((hrdata > rol_mean_adj))[0]
    peaksy = hrdata[peaksx]
    peakedges = np.where(np.diff(np.append(0, peaksx))>1)[0]
    if peaksx[-1] < len(hrdata)-1:
        peakedges = np.append(peakedges, len(peaksx))
    peak_list = []                                               
    for i in range(0, len(peakedges)-1):
        y_values = peaksy[peakedges[i]:peakedges[i+1]]
        peak_list.append(peaksx[peakedges[i] + np.argmax(y_values)])  
    peak_height = hrdata[peak_list]-rol_mean_adj[peak_list]
    # plt.plot(hrdata,'.-')
    # plt.plot(rol_mean_adj,'.-')
    # plt.plot(peak_list, hrdata[peak_list], 'x')
    return peak_list, peak_height

def find_peak_trains(peak_next, start, train=[]):
    train = train + [start]
    if not peak_next[start]:
        return [train]
    trains = []
    for node in peak_next[start]:
        newtrains = find_peak_trains(peak_next, node, train)
        for newtrain in newtrains:
            trains.append(newtrain)
    return trains

def find_two_good_peaks(peak_score_final):
    score_rv = peak_score_final[::-1]
    peaks_logic = (score_rv[:-1]<=1) & (score_rv[1:]<=1)
    if np.where(peaks_logic)[0].size > 0:
        return np.where(peaks_logic)[0][0]
    else:
        return -1

def find_best_peaks(hrdata, best_peak_para):
    win_size, adj_per, sample_rate, dist_low, dist_hi, peak_cut, rrsd_thre, shift_len = best_peak_para    
    peak_list, peak_height = find_candidate_peaks(hrdata, [win_size, sample_rate, adj_per])

    peak_list_rm, peak_height_rm = rm_low_peaks(peak_list, peak_height, [dist_low, peak_cut])
    while len(peak_list_rm) < len(peak_list):        
        peak_list = peak_list_rm
        peak_height = peak_height_rm
        peak_list_rm, peak_height_rm = rm_low_peaks(peak_list, peak_height, [dist_low, peak_cut])
    
    peak_list = np.array(peak_list)

    peak_next, peak_score = find_next_peak(peak_list, [dist_low, dist_hi]) 
    peak_trains = []
    for i in range(peak_score[0]):
        peak_trains.extend(find_peak_trains(peak_next, i, train=[]))
    trains_len = [len(train) for train in peak_trains]#list(map(len, peak_trains))
    # trains_rr_mean = list(map(lambda x: np.mean(np.diff(peak_list[x])), peak_trains))
    trains_rrsd = np.fromiter(map(lambda x: np.std(np.diff(peak_list[x])) if len(x)>1 else 0, peak_trains), dtype=float)
    trains_coverage = np.fromiter(map(lambda x: peak_list[x[-1]]-peak_list[x[0]], peak_trains), dtype=float)
    trains_unilen = np.flip(np.unique(trains_len))
    if len(peak_trains) == 0:
        next_seg_start = -1
        peak_score_final = []
        peak_height_final = []
        peak_list_final = []
    else:        
        for t_len in trains_unilen:
            temp_idx = np.where(trains_len == t_len)[0]
            min_idx = np.argmin(trains_rrsd[temp_idx])
            if trains_coverage[temp_idx[min_idx]] <= shift_len:
                next_seg_start = -1
                peak_score_final = []
                peak_height_final = []
                peak_list_final = []
                break
            else:
                if trains_rrsd[temp_idx[min_idx]] <= rrsd_thre:
                    peak_train_final = peak_trains[temp_idx[min_idx]]
                    peak_score_final = peak_score[peak_train_final]
                    peak_height_final = peak_height[peak_train_final]
                    peak_list_final = peak_list[peak_train_final]
                    rv_good_peaks = find_two_good_peaks(peak_score_final)
                    if -0.5 < rv_good_peaks < len(peak_score_final)/2:
                        next_seg_start = rv_good_peaks+2
                    else: 
                        next_seg_start = np.nan
                    break
    try:        
        return peak_list_final, peak_height_final, peak_score_final, next_seg_start
    except UnboundLocalError:
        next_seg_start = -1
        peak_score_final = []
        peak_height_final = []
        peak_list_final = []      
        for t_len in trains_unilen:
            temp_idx = np.where(trains_len == t_len)[0]
            min_idx = np.argmin(trains_rrsd[temp_idx])
            peak_train_temp = peak_trains[temp_idx[min_idx]]
            peak_score_temp = peak_score[peak_train_temp]
            rrsd_excess = trains_rrsd[temp_idx[min_idx]]/rrsd_thre
            quality_cond = (np.count_nonzero(peak_score_temp>1) < len(peak_score_temp)/5/rrsd_excess) and\
                trains_coverage[temp_idx[min_idx]] > len(hrdata)*0.75
            if (rrsd_excess < 2) and quality_cond:
                peak_train_final = peak_trains[temp_idx[min_idx]]
                peak_score_final = peak_score[peak_train_final]
                peak_height_final = peak_height[peak_train_final]
                peak_list_final = peak_list[peak_train_final]
                rv_good_peaks = find_two_good_peaks(peak_score_final)
                if -0.5 < rv_good_peaks < len(peak_score_final)/2:
                    next_seg_start = rv_good_peaks+2
                else: 
                    next_seg_start = np.nan
                break
        return peak_list_final, peak_height_final, peak_score_final, next_seg_start

    
