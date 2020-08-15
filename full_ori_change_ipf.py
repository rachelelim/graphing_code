#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:09:54 2020

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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from scipy.linalg import matfuncs

from hexrd import matrixutil as mutil


#%%

filepath = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/drift_fix/filtered_grains_files2/' #assumes grains files all in one folder
IDs = os.listdir(filepath)

#organizes the scan numbers
scan_ID1 = np.arange(21,112,4).astype(np.int8)
scan_ID2 = np.arange(22,112,4).astype(np.int8)
scan_ID3 = np.arange(23,112,4).astype(np.int8)
scans = np.concatenate([scan_ID1,scan_ID2,scan_ID3])
scanIDs = np.sort(scans)
num_scans = len(scans)
no_load_cycles = np.sort(np.concatenate([[0,1,1,2,4,5],np.arange(10,201,10),np.arange(9,201,10)]))
loaded_cycles = np.sort(np.concatenate([[0.5,1.5,4.5],np.arange(9.5,201,10)]))

no_load_scans = np.sort(np.concatenate([scan_ID1,scan_ID3]))
loaded_scans = np.sort(scan_ID2)


#%%


def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


def AxAng2Rodrigues(axis,angle):
    rod = axis * np.tan(angle/2)
    return rod


def stereographic_zproj(uvecs, cob=None):
    """
    This function takes a vstacked list of unit vectors (n, 3) and does the
    stereographic projection along a local Z axis.  If you need to project
    along a different axis, use the cob kwarg as a change-of-basis matrix.

    e.g. to project along X, cob = rot(-90, Z)*rot(-90, Y)
    """
    uvecs = np.atleast_2d(uvecs)
    npts_s = len(uvecs)
    if cob is not None:
        uvecs = np.dot(uvecs, cob.T)
    ppts = np.vstack([
            uvecs[:, 0]/(1. + uvecs[:, 2]),
            uvecs[:, 1]/(1. + uvecs[:, 2]),
            np.zeros(npts_s)
            ]).T
    return ppts


def contained_in_sst(vectors, vertices_ccw):
    """
    checks hstack array of unit vectors

    !!! inputs must both be unit vectors
    !!! vertices must be CCW
    """
    # column-wise normals to the spherical triangle edges
    sst_normals = np.array(
        [np.cross(vertices_ccw[:, i[0]],
                  vertices_ccw[:, i[1]])
         for i in [(0, 1), (1, 2), (2, 0)]
         ]
    ).T
    sst_normals_unit = mutil.unitVector(sst_normals)

    angles = np.arcsin(mutil.columnNorm(sst_normals))
    edges = []
    for i, ang in enumerate(angles):
        sub_ang = np.linspace(0, ang, endpoint=True)
        for j in sub_ang:
            rm = rot.rotMatOfExpMap(j*sst_normals_unit[:, i].reshape(3, 1))
            edges.append(np.dot(vertices_ccw[:, i], rm.T))
        edges.append(np.nan*np.ones(3))

    dim, n = vectors.shape
    contained = []
    for v in vectors.T:
        d0 = np.dot(sst_normals_unit[:, 0], v)
        d1 = np.dot(sst_normals_unit[:, 1], v)
        d2 = np.dot(sst_normals_unit[:, 2], v)
        contained.append(
            np.all([d0 > 0, d1 > 0, d2 > 0])
        )
    return contained, np.vstack(edges)

#%% Initialize slip system stuff

mat_file = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/materials2.hexrd'

pd = load_pdata(mat_file,'ti7al')

#%% Load data

data={}
stress_data = {}
grain_data = {}
num_slip_systems = 24
RSS=np.zeros([502,num_slip_systems])
RSS_all = np.zeros([len(scanIDs),len(RSS),24])
RSS_dict = {}

for i in range(len(scanIDs)): #puts all of the data from the FF into dictionaries
    grain_fname = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/drift_fix/filtered_grains_files2/filtered_scan%d_grains.out' %scanIDs[i]
    stress_fname = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/drift_fix/filtered_stress_data2/scan%d_stress.cpl' %scanIDs[i]

    grain_data['scan%d' %scanIDs[i]] = np.loadtxt(grain_fname)
    stress_data['scan%d' %scanIDs[i]] = cpl.load(open(stress_fname))

    RSS_all[i,:,:] = stress_data['scan%d' %scanIDs[i]]['RSS'] #RSS

ngrains = len(RSS)


#==============================================================================
# %%  Make color list
#==============================================================================


loaded_colors=np.arange(0,len(loaded_scans))
unloaded_colors =np.arange(0,len(no_load_scans))

loaded_colornums = np.arange(10,256,246/len(loaded_colors))
unloaded_colornums = np.arange(10,256,246/len(unloaded_colors))

cmap = plt.cm.inferno_r

cmaplist = [cmap(i) for i in range(cmap.N)]


#%%

symmetry_symbol = 'd6h' #cubic high, d6h for hex high
crystal_symmetry = sym.quatOfLaueGroup(symmetry_symbol)

# grain = 97
# grain_id = int(grain_data['scan21'][grain,0])

# mis_ang = []
# rotmats_grain = []
# expmaps_grain = []
# Eulers_grain = np.zeros([len(no_load_scans),3])

quats_init = rot.quatOfExpMap(grain_data['scan21'][:,3:6].T)

expmaps_init = grain_data['scan21'][:,3:6].T

quats_final = rot.quatOfExpMap((grain_data['scan%d'%no_load_scans[-1]][:,3:6].T))

expmaps_final = grain_data['scan111'][:,3:6].T

grainlist = []
for i in range(ngrains):
    mis = rot.misorientation(quats_init[:,i:i+1],quats_final[:,i:i+1],(crystal_symmetry,))[0]
    mis_deg = np.degrees(mis)
    if mis_deg > 0.05:
        grainlist.append(i)



#%%
rmats_init = rot.rotMatOfExpMap(np.array(expmaps_init)[:,grainlist])
rmats_final = rot.rotMatOfExpMap(np.array(expmaps_final)[:,grainlist])

# getting symmetry group directly here; could also grab from planeData object
# qsym = symm.quatOfLaueGroup('d6h')
plane_data = load_pdata('/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/materials2.hexrd', 'ti7al')
qsym = plane_data.getQSym()
bmat = plane_data.latVecOps['B']

# this was for 001 triangle
# sst_vertices = mutil.unitVector(matfuncs.triu(np.ones((3, 3))).T)
# this if for the standard triangle
# sst_vertices = mutil.unitVector(matfuncs.triu(np.ones((3, 3))))
sst_vertices = mutil.unitVector(
        np.dot(bmat,
               np.array([[0, 0, 1],
                         [2, -1, 0],
                         [1, 0, 0]]).T
        )
)
sst_normals = np.array(
    [np.cross(sst_vertices[:, i[0]], sst_vertices[:, i[1]])
     for i in [(0, 1), (1, 2), (2, 0)]
     ]
).T

# DEFINE SAMPLE DIRECTION YOU WANT FOR IPF THEN MOVE THE CRYSTAL FRAME
s = mutil.unitVector(np.c_[0., 1., 0.].T)    # sample Y a.k.a. loading
s_c_init = np.zeros((len(rmats_init), 3))
for i, rm in enumerate(rmats_init):
    s_c_init[i] = np.dot(rm.T, s).flatten()

# APPLY SYMMETRIES AND PICK REPRESENTATIVE IN SPECIFED SST
sst_list = []
csym_init = []
for c in s_c_init:
    csym_init.append(sym.applySym(c.reshape(3, 1), qsym, csFlag=True))
csym_init = np.hstack(csym_init)

sst_idx_init, edges_init = contained_in_sst(csym_init, sst_vertices)
sst_list_init = csym_init[:, np.where(sst_idx_init)[0]].T



# DEFINE SAMPLE DIRECTION YOU WANT FOR IPF THEN MOVE THE CRYSTAL FRAME
s = mutil.unitVector(np.c_[0., 1., 0.].T)    # sample Y a.k.a. loading
s_c_final = np.zeros((len(rmats_final), 3))
for i, rm in enumerate(rmats_final):
    s_c_final[i] = np.dot(rm.T, s).flatten()

# APPLY SYMMETRIES AND PICK REPRESENTATIVE IN SPECIFED SST
sst_list = []
csym_final = []
for c in s_c_final:
    csym_final.append(sym.applySym(c.reshape(3, 1), qsym, csFlag=True))
csym_final = np.hstack(csym_final)

sst_idx_final, edges_final = contained_in_sst(csym_final, sst_vertices)
sst_list_final = csym_final[:, np.where(sst_idx_final)[0]].T




#%%

cob = np.eye(3)

sst_list_t = np.dot(sst_list_init, cob.T)
edges_t = np.dot(edges_init, cob.T)



# the stereographic SST plot
fig2, ax2 = plt.subplots(figsize=(12,7))

pts_init = stereographic_zproj(sst_list_init, cob=cob)
pts_final = stereographic_zproj(sst_list_final, cob=cob)
delta_pts = (pts_final -pts_init)*10

for i in range(len(pts_init)):
    dx = delta_pts[i,0]
    dy=  delta_pts[i,1]
    if dy<0 and np.abs(dy)>np.abs(0.75*dx):
        c='b'
    elif dx<0 and np.abs(dx)>np.abs(dy):
        c='k'
    elif dx > dy and dx > 0:
        c = 'r'
    elif dy > dx and dy > 0:
        c = 'g'
    else:
        c = 'k'
    plt.arrow(pts_init[i, 0], pts_init[i, 1],dx,dy,head_width=0.5e-2,head_length=0.5e-2,color=c)
    # ax2.plot(pts[i, 0], pts[i, 1], 'o',color = 'r')
edg = stereographic_zproj(edges_init, cob=cob)
ax2.plot(edg[:, 0], edg[:, 1], 'k-')
vtx = stereographic_zproj(sst_vertices.T, cob=cob)
ax2.plot(vtx[0, 0], vtx[0, 1], 'rs',markersize=15)
ax2.plot(vtx[1, 0], vtx[1, 1], 'gs',markersize=15)
ax2.plot(vtx[2, 0], vtx[2, 1], 'bs',markersize=15)



plt.axis('equal')
plt.axis('off')
# plt.title('Grain %d \n'%grain_id+r' $\Delta$g = %.3f$^\circ$'%(mis_ang_deg[-1]),fontsize=22)

# plt.title(r' $\Delta$g = %.3f$^\circ$'%(mis_ang_deg[-1]),fontsize=22)
plt.text(-0.04,-0.05,'(0001)',fontsize=16*1.5)
plt.text(0.95,-0.05,r'(11$\bar{2}$0)',fontsize=16*1.5)
plt.text(0.8,0.53,r'(10$\bar{1}$0)',fontsize=16*1.5)


