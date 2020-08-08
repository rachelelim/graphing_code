#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:15:05 2020

@author: rachellim
"""


import numpy as np
import os
import cPickle as cpl
from matplotlib import pyplot as plt

from hexrd.xrd import rotations as rot #in python3 from hexrd import rotations as rot
#%%

main_dir = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20'
sample = 'fd1-q-1'
grains_files = '_filtered/filtered_scan_%04d_grains.out'
stress_files = '_filtered_stress/scan_%04d_stress.cpl'


scanIDs = np.arange(12,68) #for fd1-q-1


ngrains = 186 #for fd1-q-1 filtered at chi2 < 1e-2

#%% Load data

data={}
stress_data = {}
grain_data = {}


RSS_all = np.zeros([len(scanIDs),ngrains,24])

for i in range(len(scanIDs)): #puts all of the data from the FF into dictionaries
    grain_fname = os.path.join(main_dir, sample + grains_files %scanIDs[i])
    stress_fname = os.path.join(main_dir, sample + stress_files %scanIDs[i])

    grain_data['scan%d' %scanIDs[i]] = np.loadtxt(grain_fname)
    stress_data['scan%d' %scanIDs[i]] = cpl.load(open(stress_fname))

    RSS_all[i,:,:] = stress_data['scan%d' %scanIDs[i]]['RSS'] #RSS


#%% Make arrays of data

vonmises = np.zeros([ngrains,len(scanIDs)])
hydrostatic = np.zeros([ngrains,len(scanIDs)])
strain11 = np.zeros([ngrains,len(scanIDs)])
stress11 = np.zeros([ngrains,len(scanIDs)])
strain22 = np.zeros([ngrains,len(scanIDs)])
stress22 = np.zeros([ngrains,len(scanIDs)])
strain33 = np.zeros([ngrains,len(scanIDs)])
stress33 = np.zeros([ngrains,len(scanIDs)])
strain23 = np.zeros([ngrains,len(scanIDs)])
stress23 = np.zeros([ngrains,len(scanIDs)])
strain13 = np.zeros([ngrains,len(scanIDs)])
stress13 = np.zeros([ngrains,len(scanIDs)])
strain12 = np.zeros([ngrains,len(scanIDs)])
stress12 = np.zeros([ngrains,len(scanIDs)])
RSS = np.zeros([len(scanIDs),ngrains,24])


for i in range(len(scanIDs)):
    vonmises[:,i] = (stress_data['scan%d'%scanIDs[i]]['von_mises'][:,0])/1e6
    hydrostatic[:,i] = (stress_data['scan%d'%scanIDs[i]]['hydrostatic'][:,0])/1e6
    RSS[i,:,:] = (stress_data['scan%d'%scanIDs[i]]['RSS'])/1e6
    stress11[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,0])/1e6
    strain11[:,i] = grain_data['scan%d'%scanIDs[i]][:,15]
    stress22[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,1])/1e6
    strain22[:,i] = grain_data['scan%d'%scanIDs[i]][:,16]
    stress33[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,2])/1e6
    strain33[:,i] = grain_data['scan%d'%scanIDs[i]][:,17]
    stress23[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,3])/1e6
    strain23[:,i] = grain_data['scan%d'%scanIDs[i]][:,18]
    stress13[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,4])/1e6
    strain13[:,i] = grain_data['scan%d'%scanIDs[i]][:,19]
    stress12[:,i] = (stress_data['scan%d'%scanIDs[i]]['stress_S'][:,5])/1e6
    strain12[:,i] = grain_data['scan%d'%scanIDs[i]][:,20]



triaxiality = hydrostatic/vonmises



#%%
prismatic = RSS[:,:,0:3]
basal = RSS[:,:,3:6]
pyramidal = RSS[:,:,6:12]
pyramidal_ca = RSS[:,:,12:]

prismatic_max = np.max(prismatic,axis=2)
basal_max = np.max(basal,axis=2)
pyramidal_max = np.max(pyramidal,axis=2)
pyramidal_ca_max = np.max(pyramidal_ca,axis=2)


#%%

delta_strain11 = np.diff(strain11)
delta_strain22 = np.diff(strain22)
delta_strain33 = np.diff(strain33)
delta_stress11 = np.diff(stress11)
delta_stress22 = np.diff(stress22)
delta_stress33 = np.diff(stress33)
delta_vonmises = np.diff(vonmises)
delta_hydrostatic = np.diff(hydrostatic)


#%% Calculate misorientation from initial orientation

mis = np.zeros([ngrains,len(scanIDs)])

quats_init = rot.quatOfExpMap(grain_data['scan%d'%scanIDs[0]][:,3:6].T)


for i in range(len(scanIDs)):
    quats_this = rot.quatOfExpMap(grain_data['scan%d'%scanIDs[i]][:,3:6].T)
    for j in range(len(quats_init.T)):
        mis[j,i] = np.degrees(rot.misorientation(quats_init[:,j:j+1], quats_this[:,j:j+1])[0])


#%% Stress reasonability check

mean_stress11 = np.mean(stress11,axis=0)
mean_stress22 = np.mean(stress22,axis=0)
mean_stress33 = np.mean(stress33,axis=0)
mean_vonmises = np.mean(vonmises,axis=0)
mean_hydrostatic = np.mean(hydrostatic,axis=0)


fig2 = plt.figure()
plt.plot(mean_stress11,'.')
plt.plot(mean_stress22,'.')
plt.plot(mean_stress33,'.')
plt.plot(mean_vonmises,'.')
plt.plot(mean_hydrostatic,'.')
plt.legend([r'$\sigma_{11}$',r'$\sigma_{22}$',r'$\sigma_{33}$',r'$\sigma_{VM}$',r'$\sigma_{H}$'])

plt.xlabel('Step',fontsize=18)
plt.ylabel(r'Average $\sigma$', fontsize=18)

#%% Strain reasonability check

mean_strain11 = np.mean(strain11,axis=0)
mean_strain22 = np.mean(strain22,axis=0)
mean_strain33 = np.mean(strain33,axis=0)


fig3 = plt.figure()
plt.plot(mean_strain11,'.')
plt.plot(mean_strain22,'.')
plt.plot(mean_strain33,'.')

plt.legend([r'$\epsilon_{11}$',r'$\epsilon_{22}$',r'$\epsilon_{33}$'])
plt.xlabel('Step',fontsize=18)
plt.ylabel(r'Average $\epsilon$', fontsize=18)

