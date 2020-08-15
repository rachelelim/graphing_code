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

# filepath = '/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/drift_fix/filtered_grains_files2/' #assumes grains files all in one folder
# IDs = os.listdir(filepath)

scanIDs = np.arange(12,68)


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
RSS=np.zeros([186,num_slip_systems])
RSS_all = np.zeros([len(scanIDs),len(RSS),24])
RSS_dict = {}

for i in range(len(scanIDs)): #puts all of the data from the FF into dictionaries
    grain_fname = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered/filtered_scan_%04d_grains.out' %scanIDs[i]
    stress_fname = '/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered_stress/scan_%04d_stress.cpl' %scanIDs[i]

    grain_data['scan%d' %scanIDs[i]] = np.loadtxt(grain_fname)
    stress_data['scan%d' %scanIDs[i]] = cpl.load(open(stress_fname))

    RSS_all[i,:,:] = stress_data['scan%d' %scanIDs[i]]['RSS'] #RSS

ngrains = len(RSS)

quats_init = rot.quatOfExpMap(grain_data['scan12'][:,3:6].T)


#==============================================================================
# %%  Make color list
#==============================================================================


colors=np.arange(0,len(scanIDs))

colornums = np.arange(10,256,246/len(scanIDs))
cmap = plt.cm.inferno_r
cmaplist = [cmap(i) for i in range(cmap.N)]



#%%

symmetry_symbol = 'd6h' #cubic high, d6h for hex high
crystal_symmetry = sym.quatOfLaueGroup(symmetry_symbol)

grain = 49
grain_id = int(grain_data['scan12'][grain,0])

mis_ang = []
rotmats_grain = []
expmaps_grain = []
Eulers_grain = np.zeros([len(scanIDs),3])

for i in range(len(scanIDs)):

    q108 =rot.quatOfExpMap(grain_data['scan%d'%scanIDs[i]][:,3:6].T)
    expmaps_grain.append(grain_data['scan%d'%scanIDs[i]][grain,3:6])
    mis = rot.misorientation(quats_init[:,grain:grain+1], q108[:,grain:grain+1],(crystal_symmetry,))[0]
    mis_ang.append(mis)

    # rotmats_grain.append(rot.rotMatOfExpMap(grain_data['scan%d'%no_load_scans[i]][grain,3:6].T))
    # Eulers_grain[i,:] = rot.angles_from_rmat_zxz(rot.rotMatOfExpMap(expmaps_grain))


mis_ang_deg = np.degrees(np.array(mis_ang))
#%%
rmats = rot.rotMatOfExpMap(np.array(expmaps_grain).T)

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
s_c = np.zeros((len(rmats), 3))
for i, rm in enumerate(rmats):
    s_c[i] = np.dot(rm.T, s).flatten()

# APPLY SYMMETRIES AND PICK REPRESENTATIVE IN SPECIFED SST
sst_list = []
csym = []
for c in s_c:
    csym.append(sym.applySym(c.reshape(3, 1), qsym, csFlag=True))
csym = np.hstack(csym)

sst_idx, edges = contained_in_sst(csym, sst_vertices)
sst_list = csym[:, np.where(sst_idx)[0]].T


# =============================================================================
# %% PLOTTING
# =============================================================================

# the proper COB matrix for the standard SST
# for cubic SST
#cob = np.dot(
#    rot.rotMatOfExpMap(-0.5*np.pi*np.c_[0, 0, 1].T),
#    rot.rotMatOfExpMap(-0.5*np.pi*np.c_[0, 1, 0].T)
#)
cob = np.eye(3)

sst_list_t = np.dot(sst_list, cob.T)
edges_t = np.dot(edges, cob.T)

# the 3-d plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(sst_list[:, 0], sst_list[:, 1], sst_list[:, 2],
            marker='o', color='k')
ax1.scatter(sst_list_t[:, 0], sst_list_t[:, 1], sst_list_t[:, 2],
            marker='o', color='r')
ax1.set_xlabel(r"$X$")
ax1.set_ylabel(r"$Y$")
ax1.set_zlabel(r"$Z$")

# the stereographic SST plot
fig2, ax2 = plt.subplots(figsize=(6,3.5))

pts = stereographic_zproj(sst_list, cob=cob)
for i in range(len(pts)):
    ax2.plot(pts[i, 0], pts[i, 1], 'o',color = cmaplist[colornums[i]])
edg = stereographic_zproj(edges, cob=cob)
ax2.plot(edg[:, 0], edg[:, 1], 'k-')
vtx = stereographic_zproj(sst_vertices.T, cob=cob)
ax2.plot(vtx[0, 0], vtx[0, 1], 'rs',markersize=10)
ax2.plot(vtx[1, 0], vtx[1, 1], 'gs',markersize=10)
ax2.plot(vtx[2, 0], vtx[2, 1], 'bs',markersize=10)



plt.axis('equal')
plt.axis('off')
# plt.title('Grain %d \n'%grain_id+r' $\Delta$g = %.3f$^\circ$'%(mis_ang_deg[-1]),fontsize=22)

plt.title(r' $\Delta$g = %.3f$^\circ$'%(mis_ang_deg[-1]),fontsize=22)
plt.text(-0.08,-0.08,'(0001)',fontsize=16)
plt.text(0.9,-0.08,r'(10$\bar{1}$0)',fontsize=16)
plt.text(0.65,0.49,r'(11$\bar{2}$0)',fontsize=16)


ax_in = plt.axes([0,0,1,1])
ip = InsetPosition(ax2, [0.25,-0.5,0.5,0.5])
ax_in.set_axes_locator(ip)



for i in range(len(pts)):
    ax_in.plot(pts[i, 0], pts[i, 1], 'o',color = cmaplist[colornums[i]])

# plt.axis([0.29,0.32,0.02,0.035])



mark_inset(ax2, ax_in, loc1=1, loc2=2, fc="none", ec='0.5')

plt.axis('equal')
ax_in.set_xticklabels('')
ax_in.set_yticklabels('')



#%%

# fig2.savefig('/Users/rachellim/Documents/Research/CHESS_Jun17/2020-08-03/figs/ipfs/grain%d_ipf.jpg'%grain_id,
#              bbox_inches='tight',dpi=300)
