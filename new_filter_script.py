#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:51:33 2018

@author: rachel
"""

import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter

#%%

filepath = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/'
sample = 'fd1-q-1'
grains_filebase = '/scan_%04d_grains.out'

scan_IDs = np.arange(12,68)

chi2_threshold = 1e-2

#%%

grains_data = {}

for i in range(len(scan_IDs)): #load grains.out data
    grains_data['scan%d' %scan_IDs[i]] = np.loadtxt(filepath + sample + grains_filebase %scan_IDs[i])

good_grains = []
for i in range(len(grains_data['scan21'])): #grain filter
    grain_flag = True
    for j in range(len(scan_IDs)):
        if grains_data['scan%d' %scan_IDs[j]][i,2] > chi2_threshold:
            grain_flag = False

    if grain_flag == True:
        good_grains.append(i)

print('# grains tracked across all states = ' + str(len(good_grains)))

#%%

collect_directory = filepath + sample + '_filtered/'
new_file = 'filtered_scan_%04d_grains.out'

for i in range(len(scan_IDs)):
    filtered_grains_data = itemgetter(good_grains)(grains_data['scan%d' %scan_IDs[i]])
    format = '\t%3d\t%0.6f\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f\t\t%0.12f'
    header = 'grain ID\t completeness\t chi2\t\t\t xi[0]\t\t\t xi[1]\t\t\t xi[2]\t\t\t tVec_c[0]\t\t tVec_c[1]\t\t tVec_c[2]\t\t vInv_s[0]\t\t vInv_s[1]\t\t vInv_s[2]\t\t vInv_s[4]*sqrt(2)\t vInv_s[5]*sqrt(2)\t vInv_s[6]*sqrt(2)\t ln(V[0,0])\t\t ln(V[1,1])\t\t ln(V[2,2])\t\t ln(V[1,2])\t\t ln(V[0,2])\t\t ln(V[0,1])'
    np.savetxt(collect_directory + new_file %scan_IDs[i], filtered_grains_data, fmt = format, header=header)