# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:59:33 2022

@author: lxie2
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, welch
from scipy.interpolate import interp1d
from scipy import ndimage
from analysis_funcs import find_best_peaks
import os.path
import subprocess
import sys
from datetime import datetime
from time import sleep
from helper_funcs import *
# import pomegranate as pom

def extend_segs(flag):
    if flag:
        segs_start.append(seg_start+v_loc)
        segs_end.append(seg_end+v_loc)
        segs_peak.append(peak_list_final.tolist())
        segs_rrsd.append(np.std(np.diff(peak_list_final)))
        segs_height.append(peak_height_final.tolist())
        segs_score.append(peak_score_final.tolist())
    else: # if flag==False, the segment terminates at shift_len
        segs_start.append(seg_start+v_loc)
        segs_end.append(seg_start+shift_len+v_loc)
        segs_peak.append([])
        segs_rrsd.append(-1)
        segs_height.append([])
        segs_score.append([])

# concatenate prev seg after removing rm_num peaks from its right end
def extend_all_peaks(rm_num):
    if rm_num == -1:
        all_peaks_pos.append(np.nan)
        all_peaks_time.append(np.nan)
        all_peaks_height.append(np.nan)
        all_peaks_score.append(np.nan)
    elif rm_num == 0:
        all_peaks_pos.extend(prev_peaks_pos.tolist())
        all_peaks_time.extend(prev_peaks_time.tolist())
        all_peaks_height.extend(prev_peaks_height.tolist())
        all_peaks_score.extend(prev_peaks_score.tolist())
    else:
        all_peaks_pos.extend(prev_peaks_pos[:-rm_num].tolist())
        all_peaks_time.extend(prev_peaks_time[:-rm_num].tolist())
        all_peaks_height.extend(prev_peaks_height[:-rm_num].tolist())
        all_peaks_score.extend(prev_peaks_score[:-rm_num].tolist())

# update the prev seg with the curr seg after removing rm_num peaks from its left end
def convert_curr_seg(rm_num):
    if rm_num == -1:
        prev_peaks_pos = np.array([], dtype=int)
        prev_peaks_time = np.array([])
        prev_peaks_height = np.array([])
        prev_peaks_score = np.array([], dtype=int)
    else:
        prev_peaks_pos = peak_list_final[rm_num:].copy()
        prev_peaks_time = peak_time_final[rm_num:].copy()
        prev_peaks_height = peak_height_final[rm_num:].copy()
        prev_peaks_score = peak_score_final[rm_num:].copy()
    return prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score


def clean_peaks_seg(nan_idx, peaks_pos_seg, peaks_time_seg):
    for i in nan_idx[::-1]:
        peaks_pos_seg.pop(i)
        peaks_time_seg.pop(i)
        tdiff = peaks_time_seg[i]-peaks_time_seg[i-1]
        peaks_time_seg[i:] = [x-tdiff for x in peaks_time_seg[i:]]
        peaks_pos_seg.pop(i)
        peaks_time_seg.pop(i)




folder_name = sys.argv[1]+'/'
max_wake = [int(x) for x in sys.argv[2].split(":")]
sample_rate = 130.
win_size = 0.75
adj_per = 95.
dist_low = sample_rate/2
dist_hi = sample_rate
peak_cut = 0.25
rrsd_thre = 15
shift_len = int(sample_rate*5)
best_peak_para = [win_size, adj_per, sample_rate, dist_low, dist_hi, peak_cut, rrsd_thre, shift_len]
def_len = int(sample_rate*20)
pcs = 1e-6
b, a = iirnotch(w0=0.05, Q = 0.005, fs = sample_rate)
with open('model8_dict.pkl', 'rb') as fp:
    dist_list, prior_params = pickle.load(fp)
bayes_clf = bayes_wrap(dist_list)

sleep_bool = True
file_num = 2
segs_start = [] # list containing the beginning of each segment
segs_end = [] # list containing the (not included) end of each segment
# list containing the # of overlapping peaks at the two joints on both ends.
# a clean segment should have 2 overlapping peaks on both ends: [2, 2]
# a discarded segment due to unidentifiable peaks would have [-1, -1].
segs_jnt = []
segs_peak = [] # list containing the peak positions of each segment
segs_score = [] # list containing the peak scores of each segment
segs_height = [] # list containing the peak heights of each segment
segs_rrsd = [] # list containing the rrsd of each segment. -1 for discarded segs

# lists for all peaks
all_peaks_pos = []
all_peaks_time = []
all_peaks_height = []
all_peaks_score = []
low_quality_peaks = []
prev_peaks_pos = np.array([])
prev_peaks_time = np.array([])
curr_seg_start = np.nan

v_loc = 0
curr_v = []
curr_t = []
b_len = 300;
peaks_start = 0
peaks_end = b_len
peaks_overlap = 100
feature_lists = []
feature_time = []
sleep_stage = []
# feature_peak_index = []
rr_count = 0;
rr_cum = 0;
time_convert = lambda x: x[0]*60+x[1]
stage_val = [3, 4, 0, 3, 1, 2]
w = 10

# v_all = []
while sleep_bool:
    now = datetime.now()
    now_list = [now.hour, now.minute]
    if (time_convert(now_list)>time_convert(max_wake)): #(now_list[0] <22) &
        print('wake')
        sleep_bool = False
        break
    if os.path.isfile(folder_name + f'd{file_num}.pkl'):
        if not os.path.isfile(folder_name + f'd{file_num+1}.pkl'):
            check_process = subprocess.run(["lsof", "-t",  f'd{file_num}.pkl'], capture_output=True)
            if len(check_process.stdout)>0:
                sleep(10)
                continue
    else:
        sleep(60)
        continue
    with open(folder_name+f'd{file_num}.pkl', 'rb') as d_file:
        v_temp = pickle.load(d_file)
    with open(folder_name+f't{file_num}.pkl', 'rb') as t_file:
        t_temp = pickle.load(t_file)
    curr_v.extend(v_temp)
    curr_t.extend(t_temp)
    t_array = np.asarray(curr_t)*1e-9
    t_idx = np.where(t_array>pcs)[0]
    t_ratio = np.diff(t_array[t_idx])/np.diff(t_idx)*130
    t_logic = (t_ratio>0.95) & (t_ratio<1.05)
    idx_starts, idx_ends = find_valid_segs(t_logic, 20)
    # this wouldn't work if no valid start can be found in the first file, I'll deal with this later
    if file_num == 2:
        t0 = t_array[t_idx[idx_starts[0]]]
        valid_end = 0
        sleep_start = datetime.fromtimestamp(os.path.getmtime(folder_name+f'd{file_num}.pkl'))
        if sleep_start.hour > 22:
            sleep_time = [sleep_start.hour-23, sleep_start.minute-60]
        else:
            sleep_time = [sleep_start.hour, sleep_start.minute]
        prior_t = (time_convert(max_wake) - time_convert(sleep_time))/60
        prior_f = prior_func(8, prior_params)
    # v_all.extend(v_temp)
    v_filtered = filtfilt(b, a, curr_v)
    v_filtered = ndimage.gaussian_filter(v_filtered, 2)
    # the start and end of the first segment from the current file
    if len(idx_starts) == 0:
        if prev_peaks_pos.size > 0: # if the previous segment exists
            extend_all_peaks(0) # concatenate without modifying the right joint
            # the previous segment is updated
            prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                convert_curr_seg(-1)
            extend_all_peaks(-1) # nan added to all_peaks_* to indicate discontinuity
            curr_seg_start = np.nan # new current segment has no right joint
    else:
        for valid_count in range(len(idx_starts)):
            if (valid_end < np.inf) | (idx_starts[valid_count]>0):
                valid_start = t_idx[idx_starts[valid_count]]
            else:
                valid_start = 0;
            if idx_ends[valid_count]<np.inf:
                valid_end = t_idx[idx_ends[valid_count]]
                time_interp = interp1d(t_idx[idx_starts[valid_count]:idx_ends[valid_count]], \
                                       t_array[t_idx[idx_starts[valid_count]:idx_ends[valid_count]]]-t0, \
                                      bounds_error=False, fill_value="extrapolate")
            else:
                valid_end = np.inf
                time_interp = interp1d(t_idx[idx_starts[valid_count]:], \
                                       t_array[t_idx[idx_starts[valid_count]:]]-t0, \
                                      bounds_error=False, fill_value="extrapolate")
            seg_start = valid_start
            if seg_start < valid_end - 2*def_len:
                seg_end = seg_start + def_len
            else:
                seg_end = valid_end

            while seg_end <= min([valid_end, v_filtered.size]):
                hrdata = v_filtered[seg_start:seg_end]
                peak_list_final, peak_height_final, peak_score_final, next_seg_start = \
                    find_best_peaks(hrdata, best_peak_para)
                if next_seg_start == -1: # curr seg invalid: only a few peaks are identified in the stretch
                    extend_segs(False) # record that this segment is invalid
                    seg_start = seg_start+shift_len # start the next segment after shift_len
                    segs_jnt.append([-1, -1])
                    if prev_peaks_pos.size > 0: # if the previous segment exists
                        extend_all_peaks(0) # concatenate without modifying the right joint
                        # the previous segment is updated
                        prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                            convert_curr_seg(-1)
                        extend_all_peaks(-1) # nan added to all_peaks_* to indicate discontinuity
                        curr_seg_start = np.nan # new current segment has no right joint
                else: # curr seg is valid
                    peak_time_final = np.array(time_interp(peak_list_final + seg_start))
                    peak_list_final = peak_list_final + seg_start + v_loc
                    segs_jnt.append([0, 0])
                    extend_segs(True)
                    if prev_peaks_pos.size > 0:
                        # curr_seg_start: the # of overlapping peaks at the end of prev_peaks_pos
                        # and beginning of current segment peak_list_final
                        if not np.isnan(curr_seg_start):
                            prev_peaks = prev_peaks_pos[-curr_seg_start:]
                            curr_peaks = peak_list_final[:curr_seg_start]
                            # peaks in the prev seg but not curr seg
                            prev_peaks_incon = np.setdiff1d(prev_peaks, curr_peaks, assume_unique=True)
                            # peaks in the curr seg but not prev seg
                            curr_peaks_incon = np.setdiff1d(curr_peaks, prev_peaks, assume_unique=True)
                            if prev_peaks_incon.size==0 and curr_peaks_incon.size==0:
                                extend_all_peaks(0)
                                prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                    convert_curr_seg(curr_seg_start)
                            elif prev_peaks_incon.size>1 or curr_peaks_incon.size>1:
                                print ("more than 1 inconsistency over the joint #"+str(len(segs_start)-1))
                                segs_jnt[-2][1] = curr_seg_start
                                segs_jnt[-1][0] = curr_seg_start
                                # concatenate the prev seg curr_seg_start without the last curr_seg_start peaks
                                extend_all_peaks(curr_seg_start)
                                extend_all_peaks(-1) # patch with a nan to indicate discontinuity
                                prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                    convert_curr_seg(curr_seg_start)
                            elif prev_peaks_incon[0] == prev_peaks[-1]: # the last peak is inconsistent
                                if curr_peaks_incon[0] != curr_peaks[-1]:
                                    raise ValueError("unexpected inconsistency around " + str(curr_peaks_incon[0]))
                                if prev_peaks[-1] < curr_peaks[-1]*0.1: # if the false peak can be identified based on height
                                    segs_jnt[-2][1] = 1
                                    extend_all_peaks(1)
                                    prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                        convert_curr_seg(curr_seg_start-1)
                                else: # construct two alternative trains, and identify the false peak based on rrsd
                                    joined_seg1 = prev_peaks_pos[:-1].tolist() + \
                                        peak_list_final[curr_seg_start-1:].tolist()
                                    joined_seg2 = prev_peaks_pos.tolist() + \
                                        peak_list_final[curr_seg_start:].tolist()
                                    joined_rrsd1 = np.std(np.diff(joined_seg1))
                                    joined_rrsd2 = np.std(np.diff(joined_seg2))
                                    if joined_rrsd1 < joined_rrsd2:
                                        segs_jnt[-2][1] = 1
                                        extend_all_peaks(1)
                                        prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                            convert_curr_seg(curr_seg_start-1)
                                    else:
                                        print ("the old segment is better at joint #"+str(len(segs_start)-1))
                                        segs_jnt[-1][0] = 1
                                        extend_all_peaks(0)
                                        prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                            convert_curr_seg(curr_seg_start)
                            else:
                                print ("the inconsistent peak is not the last one at joint #"+str(len(segs_start)-1))
                                segs_jnt[-2][1] = curr_seg_start
                                segs_jnt[-1][0] = curr_seg_start
                                extend_all_peaks(curr_seg_start)
                                extend_all_peaks(-1)
                                prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                    convert_curr_seg(curr_seg_start)
                        else: # np.isnan(curr_seg_start) is true
                            extend_all_peaks(0)
                            extend_all_peaks(-1)
                            prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                                convert_curr_seg(0)
                    else: # prev seg invalid but curr seg valid
                        prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                            convert_curr_seg(0)
                    curr_seg_start = next_seg_start
                    if np.isnan(curr_seg_start):
                        seg_start = prev_peaks_pos[-1] + 50 - v_loc
                        segs_end[-1] = prev_peaks_pos[-1] + 50 # maybe not
                    elif curr_seg_start == -1:
                        raise ValueError("curr_seg_start is -1")
                    else:
                        seg_start = prev_peaks_pos[-curr_seg_start] - v_loc - 50
                if seg_end < valid_end:
                    if seg_start < valid_end - 2*def_len:
                        seg_end = seg_start + def_len
                    else:
                        seg_end = valid_end
                else:
                    if prev_peaks_pos.size > 0: # if the previous segment exists
                        extend_all_peaks(0) # concatenate without modifying the right joint
                        # the previous segment is updated
                        prev_peaks_pos, prev_peaks_time, prev_peaks_height, prev_peaks_score = \
                            convert_curr_seg(-1)
                        extend_all_peaks(-1) # nan added to all_peaks_* to indicate discontinuity
                        curr_seg_start = np.nan # new current segment has no right joint
                    seg_end = valid_end+1
        if valid_end < np.inf:
            curr_v = []
            curr_t = []
            if prev_peaks_pos.size > 0: # if the previous segment exists
                raise ValueError("prev_peaks_pos should have size 0")
        else:
            curr_v = curr_v[seg_start:]
            curr_t = curr_t[seg_start:]
            v_loc = v_loc + seg_start
    file_num += 1
    while peaks_end < len(all_peaks_time):
        if np.isnan(all_peaks_pos[peaks_start]):
            peaks_start = peaks_start+1
        if np.isnan(all_peaks_score[peaks_end-1]):
            peaks_end = peaks_end-1
        peaks_pos_seg = all_peaks_pos[peaks_start:peaks_end].copy()
        peaks_time_seg = all_peaks_time[peaks_start:peaks_end].copy()
        nan_idx = np.where(np.isnan(peaks_pos_seg))[0]
        if nan_idx.size > 0:
            clean_peaks_seg(nan_idx, peaks_pos_seg, peaks_time_seg)
        if np.where(np.diff(peaks_time_seg)>3)[0].size>0:
            print(f'RR interval larger than 3s {peaks_start}, {peaks_end}')
            feature_lists.append(feature_lists[-1])
        else:
            vlf, lf, hf = freq_analysis(peaks_time_seg)
            mean_rr, sdnn, pnn20, pnn30, pnn40 = time_analysis(peaks_time_seg)
            feature_lists.append([np.log(lf), np.log(hf), lf/hf, mean_rr, sdnn, pnn20])
            # rr_cum = rr_cum + (len(peaks_time_seg)-1)*mean_rr
            # rr_count = rr_count+len(peaks_time_seg)-1
        feature_time.append(peaks_time_seg[-1])
        prior = prior_f(feature_time[-1])
        # clf = pom.BayesClassifier(dists, prior)
        # stage_predict = clf.predict(feature_lists[-1])
        stage_predict = bayes_clf(feature_lists[-1], prior)
        print(stage_predict)
        # feature_peak_index.append((peaks_start, peaks_end))
        sleep_stage.append(stage_val[stage_predict])
        peaks_start = peaks_end-peaks_overlap
        peaks_end = peaks_start+b_len
    stage_binary = [1. if x>2.5 else 0. for x in sleep_stage]
    if on_switch(stage_binary, feature_time, w):
        print('wake')
        sleep_bool = False
        break
