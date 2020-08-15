#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:10:14 2020

@author: rachellim
"""


import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.patches as mpatches
# import probscale as pb
import cPickle as cpl
from hexrd.xrd import rotations as rot
from hexrd import matrixutil as mutil
from hexrd.xrd import symmetry   as sym
import copy

from scipy.stats import probplot
from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D


#%%

grains_files = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered/filtered_scan_%04d_grains.out'
stress_files = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered_stress/scan_%04d_stress.cpl'

scanIDs = np.arange(12,68)




#%% Helper functions

def filter_grains(grainsout,chi2_thresh=1e-2,completeness=0.95,hthresh=0.6):
    good_grains = np.where((grainsout[:,2]<chi2_thresh) & (grainsout[:,1]>completeness) &
                                          (grainsout[:,6] < hthresh) & (grainsout[:,6] > -hthresh) &
                                          (grainsout[:,8] < hthresh) & (grainsout[:,8] > -hthresh))
    return good_grains[0]


def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData





#%% Initialize slip system stuff

mat_file = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/materials2.hexrd'

pd = load_pdata(mat_file,'ti7al')

ngrains = 186

#%% Load data

data={}
stress_data = {}
grain_data = {}


RSS_all = np.zeros([len(scanIDs),ngrains,24])

for i in range(len(scanIDs)): #puts all of the data from the FF into dictionaries
    grain_fname = grains_files %scanIDs[i]
    stress_fname = stress_files %scanIDs[i]

    grain_data['scan%d' %scanIDs[i]] = np.loadtxt(grain_fname)
    stress_data['scan%d' %scanIDs[i]] = cpl.load(open(stress_fname))

    RSS_all[i,:,:] = stress_data['scan%d' %scanIDs[i]]['RSS'] #RSS





#%%

COM = grain_data['scan%d'%scanIDs[0]][:,6:9]


#%% Calculate the different scalars of interest

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

delta_strain11 = np.diff(strain11)
delta_strain22 = np.diff(strain22)
delta_strain33 = np.diff(strain33)
delta_stress11 = np.diff(stress11)
delta_stress22 = np.diff(stress22)
delta_stress33 = np.diff(stress33)
delta_vonmises = np.diff(vonmises)
delta_hydrostatic = np.diff(hydrostatic)

#%%
prismatic = RSS[:,:,0:3]
basal = RSS[:,:,3:6]
pyramidal = RSS[:,:,6:12]
pyramidal_ca = RSS[:,:,12:]

prismatic_max = np.max(prismatic,axis=2)
basal_max = np.max(basal,axis=2)
pyramidal_max = np.max(pyramidal,axis=2)
pyramidal_ca_max = np.max(pyramidal_ca,axis=2)



# del_eps22_cyc1 = grain_data['scan23'][:,17] - grain_data['scan21'][:,17]
# del_eps22_cyc2 = grain_data['scan25'][:,17] - grain_data['scan23'][:,17]
# del_eps22_cyc5 = grain_data['scan31'][:,17] - grain_data['scan25'][:,17]


mis = np.zeros([ngrains,len(scanIDs)])

quats_init = rot.quatOfExpMap(grain_data['scan%d'%scanIDs[0]][:,3:6].T)


for i in range(len(scanIDs)):
    quats_this = rot.quatOfExpMap(grain_data['scan%d'%scanIDs[i]][:,3:6].T)
    for j in range(len(quats_init.T)):
        mis[j,i] = np.degrees(rot.misorientation(quats_init[:,j:j+1], quats_this[:,j:j+1])[0])




#%%
sig_macro = np.array([0,540,0,0,0,0])
mag_sig_macro = np.linalg.norm(sig_macro)

coax = np.zeros([ngrains,len(scanIDs)])

for scan in range(4,27):#len(scanIDs)):
    print(scan)

    sig_grain = stress_data['scan%d'%scanIDs[scan]]['stress_S']
    for grains in range(0,len(sig_grain)):
        # sig_macro = stress_data['scan%d'%scanIDs[0]]['stress_S'][grains,:]
        # mag_sig_macro = np.linalg.norm(sig_macro)
        mag_sig_grain = np.linalg.norm(sig_grain[grains,:])
        coaxiality = np.degrees(np.arccos(np.inner(sig_macro,sig_grain[grains,:])/(mag_sig_grain*mag_sig_macro)))
        coax [grains,scan] = coaxiality



#==============================================================================
# %%  Make color list
#==============================================================================

colors=np.arange(0,len(scanIDs))

colornums = np.arange(10,256,246/len(scanIDs))
cmap = plt.cm.inferno_r
cmaplist = [cmap(i) for i in range(cmap.N)]



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


#%% stress22 Location

strain_lim = 3e-3

figY = plt.figure(figsize=(5,4))

# ax1 = plt.subplot(1,2,1)
plt.scatter(COM[:,0],COM[:,2],s=100,c=strain22[:,0],alpha=1,vmin=-strain_lim,vmax=strain_lim,cmap='bwr')
plt.title('Initial',fontsize=18)
plt.axis('off')

# ax2 = plt.subplot(1,2,2)
# plt.scatter(COM[:,0],COM[:,2],s=50,c=strain22[:,1],alpha=1,vmin=-strain_lim,vmax=strain_lim,cmap='bwr')
# plt.title('Cycle 1',fontsize=18)
# plt.axis('off')

# figY.subplots_adjust(right=0.05)


# cax = plt.axes([0.15, -0.05, 0.73, 0.06])
cbar = plt.colorbar()
cbar.set_ticks([-3e-3,-2e-3,-1e-3,0,1e-3,2e-3,3e-3])
cbar.set_label(r'$\epsilon_{yy}$', FontSize=18)

#==============================================================================
# %%  Von Mises Heatmap
#==============================================================================


fig1 = plt.figure(figsize=(8,8))
plt.imshow(delta_vonmises,cmap='bwr',extent = (0,56,00,56),vmin=-600,vmax=600)

cbar = plt.colorbar()
cbar.set_label(r'$\Delta$ $\sigma_{VM}$ (MPa)', FontSize=18)


#==============================================================================
# %%  Misorientation Probplot
#==============================================================================

fig4 = plt.figure()
ax1 = plt.subplot(111)


for i in range(1,mis.shape[1]):
    pp = mis[:,i]
    x,y= probplot(pp)[0]
    plt.plot(y,x,'.',color=cmaplist[colornums[i]])


plt.xlim(left=0,right=0.8)

# # ax.tick_params(labelsize=14)

red_patch = mpatches.Patch(color=cmaplist[colornums[1]], label='Initial') #Labels
blue_patch = mpatches.Patch(color=cmaplist[colornums[i]], label='Final')

plt.legend(handles=[red_patch,blue_patch],fontsize=12,loc='lower right')
plt.xlabel(r'$\Delta$ orientation ($^\circ$)',fontsize=18)
plt.ylabel('Quantile',fontsize=18)


#==============================================================================
# %%  Von Mises location
#==============================================================================

vm_max = 600
vm_min = 200

fig5 = plt.figure(figsize=(6,5))

for i in range(len(scanIDs)):
    plt.clf()
    plt.figure(figsize=(6,5))
    plt.scatter(COM[:,0],COM[:,2],c=vonmises[:,i],s=vonmises[:,i],alpha=0.75,
                vmin=vm_min,vmax=vm_max,cmap='Reds',edgecolors='k',linewidths=0.5)
    # plt.axis('off')

    cbar = plt.colorbar()
    cbar.set_label(r'$\sigma_{VM}$ (MPa)', FontSize=18)

    plt.xlim(left=-0.5,right=0.5)
    plt.ylim(bottom=-0.5,top=0.5)

    plt.savefig('/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_analysis/vMstress_location/scan_%04d.jpg'%scanIDs[i],
                dpi=300,bbox_inches='tight')
    plt.close()




#%%


vm_max = 600
vm_min = 200



for i in range(len(scanIDs)):
    plt.clf()
    figAA = plt.figure(figsize=(6,5))
    ax = figAA.add_subplot(111, projection='3d')
    ax.scatter(COM[:,0],COM[:,1],COM[:,2],s=vonmises[:,i]/2,c=vonmises[:,i],alpha=0.75,
               vmin=vm_min,vmax=vm_max,cmap='Reds',edgecolors='k',linewidths=0.5)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\sigma_{VM}$ (MPa)', FontSize=18)
    # plt.savefig('/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_analysis/vMstress_location_3d/scan_%04d.jpg'%scanIDs[i],
    #             dpi=300,bbox_inches='tight')
    # plt.close()

#==============================================================================
# %%  Delta Von Mises location
#==============================================================================

vm_max = 500
vm_min = -500

fig5 = plt.figure(figsize=(6,5))

for i in range(len(scanIDs)-1):
    plt.clf()
    plt.figure(figsize=(6,5))
    plt.scatter(COM[:,0],COM[:,2],c=delta_vonmises[:,i],alpha=0.75,s=(delta_vonmises[:,i]+600)/2,
                vmin=vm_min,vmax=vm_max,cmap='bwr',edgecolors='k',linewidths=0.5)
    # plt.axis('off')

    cbar = plt.colorbar()
    cbar.set_label(r'$\Delta$ $\sigma_{VM}$ (MPa)', FontSize=18)

    plt.xlim(left=-0.5,right=0.5)
    plt.ylim(bottom=-0.5,top=0.5)

    plt.savefig('/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_analysis/delta_vMstress_location/scan_%04d.jpg'%scanIDs[i+1],
                dpi=300,bbox_inches='tight')
    plt.close()

#%%


vm_max = 500
vm_min = -500



for i in range(len(scanIDs)):
    plt.clf()
    figBB = plt.figure(figsize=(6,5))
    ax1 = figBB.add_subplot(111, projection='3d')
    ax1.scatter(COM[:,0],COM[:,1],COM[:,2],c=delta_vonmises[:,i],alpha=0.75,s=(delta_vonmises[:,i]+600)/3,
               vmin=vm_min,vmax=vm_max,cmap='bwr',edgecolors='k',linewidths=0.5)
    # ax1.colorbar()
    # cbar.set_label(r'$\sigma_{VM}$ (MPa)', FontSize=18)
    # plt.savefig('/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_analysis/delta_vMstress_location_3d/scan_%04d.jpg'%scanIDs[i],
    #             dpi=300,bbox_inches='tight')
    # plt.close()


#==============================================================================
# %%  Delta Von Mises analysis
#==============================================================================

del_vm_thresh = -150

events= (delta_vonmises < del_vm_thresh)


#==============================================================================
# %%  RSS stuff may be interesting
#==============================================================================

basal_events = basal_max[:-1,:].T[events]
prism_events = prismatic_max[:-1,:].T[events]
vm_events =np.abs(delta_vonmises[events])

fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.scatter(prism_events,basal_events,np.abs(vm_events),c=basal_events,s=50)
ax6.set_ylabel('basal')
ax6.set_xlabel('prism')
ax6.set_zlabel('vm')

#%%

fig7 = plt.figure()

plt.plot(basal_events,vm_events,'.')

# plt.xlim(left=0,right=350)
# plt.ylim(bottom=125,top=350)
plt.xlabel('Max Basal RSS')
plt.ylabel(r'Subsequent magnitude of drop in $\sigma_{VM}$')


fig8 = plt.figure()

plt.plot(prism_events,vm_events,'.')

plt.xlim(left=0,right=350)
plt.ylim(bottom=125,top=350)
plt.xlabel('Max Prismatic RSS')
plt.ylabel(r'Subsequent magnitude of drop in $\sigma_{VM}$')
