#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:27:04 2020

@author: rachellim
"""


import numpy as np
# from hexrd_shared.scans import ScanSet
# from expmap import ExpMap
# from utils import to3x3, change_basis

# for symmetries
# import crystalsymmetry, quaternions

from matplotlib import pyplot as plt

from hexrd.xrd import symmetry   as sym
from hexrd.xrd import rotations as rot
from hexrd import matrixutil as mutil
from hexrd import config

import cPickle as cpl

#%%

def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


def calc_moduli(T_in_Celsius):
    #Fischer  Values
#    T_in_Celsius=XXX
    C11_List=np.array([1.761,1.759,1.749,1.726,1.699,1.668,1.639,1.624,1.609,1.579,1.551,1.522,1.495,1.468,1.442,1.416,1.392,1.368,1.345,1.322,1.299,1.276,1.253,1.231,1.196])*1e11
    C33_List=np.array([1.905,1.905,1.894,1.876,1.857,1.837,1.816,1.807,1.795,1.775,1.753,1.734,1.715,1.696,1.678,1.661,1.644,1.627,1.610,1.593,1.576,1.560,1.545,1.529,1.504])*1e11
    C44_List=np.array([0.508,0.508,0.505,0.499,0.490,0.481,0.472,0.467,0.462,0.453,0.444,0.434,0.424,0.414,0.403,0.392,0.381,0.370,0.359,0.348,0.337,0.326,0.316,0.307,0.291])*1e11
    C13_List=np.array([0.683,0.682,0.680,0.681,0.684,0.687,0.689,0.690,0.681,0.694,0.695,0.695,0.682,0.692,0.691,0.690,0.692,0.688,0.688,0.688,0.688,0.688,0.688,0.688,0.688])*1e11
    C12_List=np.array([0.869,0.867,0.871,0.877,0.889,0.901,0.913,0.920,0.925,0.934,0.943,0.952,0.961,0.967,0.973,0.978,0.983,0.985,0.988,0.991,0.992,0.993,0.994,0.996,0.996])*1e11
    T_Stiff=np.array([4,23,73,123,173,223,273,298,323,373,423,473,523,573,623,673,723,773,823,873,923,973,1023,1073,1156]).astype(float)-273.15

    C11=np.interp(T_in_Celsius,T_Stiff,C11_List)
    C33=np.interp(T_in_Celsius,T_Stiff,C33_List)
    C13=np.interp(T_in_Celsius,T_Stiff,C13_List)
    C12=np.interp(T_in_Celsius,T_Stiff,C12_List)
    C44=np.interp(T_in_Celsius,T_Stiff,C44_List)

    c_mat_C=np.array([[C11,C12,C13,0.,0.,0.],
    				  [C12,C11,C13,0.,0.,0.],
    				  [C13,C13,C33,0.,0.,0.],
    				  [0.,0.,0.,C44,0.,0.],
    				  [0.,0.,0.,0.,C44,0.],
    				  [0.,0.,0.,0.,0.,0.5*(C11-C12)]])

    return c_mat_C


def post_process_stress(grain_data,c_mat_C,schmid_T_list=None):
    num_grains=grain_data.shape[0]

    stress_S=np.zeros([num_grains,6])
    stress_C=np.zeros([num_grains,6])
    hydrostatic=np.zeros([num_grains,1])
    pressure=np.zeros([num_grains,1])
    von_mises=np.zeros([num_grains,1])

    if schmid_T_list is not None:
        num_slip_systems=schmid_T_list.shape[0]
        RSS=np.zeros([num_grains,num_slip_systems])


    for jj in np.arange(num_grains):

        expMap=np.atleast_2d(grain_data[jj,3:6]).T
        strainTmp=np.atleast_2d(grain_data[jj,15:21]).T

        #Turn exponential map into an orientation matrix
        Rsc=rot.rotMatOfExpMap(expMap)

        strainTenS = np.zeros((3, 3), dtype='float64')
        strainTenS[0, 0] = strainTmp[0]
        strainTenS[1, 1] = strainTmp[1]
        strainTenS[2, 2] = strainTmp[2]
        strainTenS[1, 2] = strainTmp[3]
        strainTenS[0, 2] = strainTmp[4]
        strainTenS[0, 1] = strainTmp[5]
        strainTenS[2, 1] = strainTmp[3]
        strainTenS[2, 0] = strainTmp[4]
        strainTenS[1, 0] = strainTmp[5]


        strainTenC=np.dot(np.dot(Rsc.T,strainTenS),Rsc)
        strainVecC = mutil.strainTenToVec(strainTenC)


        #Calculate stress
        stressVecC=np.dot(c_mat_C,strainVecC)
        stressTenC = mutil.stressVecToTen(stressVecC)
        stressTenS = np.dot(np.dot(Rsc,stressTenC),Rsc.T)
        stressVecS = mutil.stressTenToVec(stressTenS)

        #Calculate hydrostatic stress
        hydrostaticStress=(stressVecS[:3].sum()/3)


        #Calculate Von Mises Stress
        devStressS=stressTenS-hydrostaticStress*np.identity(3)
        vonMisesStress=np.sqrt((3/2)*(devStressS**2).sum())


        #Project on to slip systems
        if schmid_T_list is not None:
            for ii in np.arange(num_slip_systems):
                RSS[jj,ii]=np.abs((stressTenC*schmid_T_list[ii,:,:]).sum())


        stress_S[jj,:]=stressVecS.flatten()
        stress_C[jj,:]=stressVecC.flatten()

        hydrostatic[jj,0]=hydrostaticStress
        pressure[jj,0]=-hydrostaticStress
        von_mises[jj,0]=vonMisesStress

    stress_data=dict()

    stress_data['stress_S']=stress_S
    stress_data['stress_C']=stress_C
    stress_data['hydrostatic']=hydrostatic
    stress_data['pressure']=pressure
    stress_data['von_mises']=von_mises

    if schmid_T_list is not None:
        stress_data['RSS']=RSS

    return stress_data


def gen_schmid_tensors(pd,uvw,hkl):

    # slip plane directions
    slipdir  = mutil.unitVector( np.dot( pd.latVecOps['F'], uvw) ) #  2 -1 -1  0
    slipdir_sym  = sym.applySym(slipdir, pd.getQSym(), csFlag=False, cullPM=True, tol=1e-08)

    # slip plane plane normals
    n_plane = mutil.unitVector( np.dot( pd.latVecOps['B'], hkl ) )
    n_plane_sym = sym.applySym(n_plane, pd.getQSym(), csFlag=False, cullPM=True, tol=1e-08)


    num_slip_plane= n_plane_sym.shape[1]

    num_slip_sys=0
    for i in range(num_slip_plane):
        planeID = np.where(abs(np.dot(n_plane_sym[:, i],slipdir_sym)) < 1.e-8)[0]
        num_slip_sys +=planeID.shape[0]

    T= np.zeros((num_slip_sys, 3, 3))
    counter=0
        #
    for i in range(num_slip_plane):
        planeID = np.where(abs(np.dot(n_plane_sym[:, i],slipdir_sym)) < 1.e-8)[0]
        for j in np.arange(planeID.shape[0]):
            T[counter, :, :] = np.dot(slipdir_sym[:, planeID[j]].reshape(3, 1), n_plane_sym[:, i].reshape(1, 3))
            counter+=1
    #Clean some round off errors
    round_off_err=np.where(abs(T)<1e-8)
    T[round_off_err[0],round_off_err[1],round_off_err[2]]=0.

    return T



#%%

mat_file = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-07-30/materials2.hexrd'
mat = 'ti7al'

pd = load_pdata(mat_file, mat)

c_mat_C = calc_moduli(25)

T=np.zeros([24,3,3])

T[0:3,:,:]=gen_schmid_tensors(pd,np.atleast_2d(np.array([1,0,0])).T,np.atleast_2d(np.array([1,0,0])).T)#prism
T[3:6,:,:]=gen_schmid_tensors(pd,np.atleast_2d(np.array([1,0,0])).T,np.atleast_2d(np.array([0,0,1])).T)#basal
T[6:12,:,:]=gen_schmid_tensors(pd,np.atleast_2d(np.array([1,0,0])).T,np.atleast_2d(np.array([1,0,1])).T)#pyr a
T[12:,:,:]=gen_schmid_tensors(pd,np.atleast_2d(np.array([1,0,1])).T,np.atleast_2d(np.array([1,0,1])).T)# pyr c+a


scan_IDs = np.arange(12,68)



stress_data = {}
fstem ='/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered/filtered_scan_%04d_grains.out'
stress_fstem = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered_stress/scan_%04d_stress.cpl'
for i in range(len(scan_IDs)):
    grain_data = np.loadtxt(fstem %scan_IDs[i])
    stress_data = post_process_stress(grain_data, c_mat_C, T)

    cpl.dump(stress_data,open(stress_fstem %scan_IDs[i],'wb'))





