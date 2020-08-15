from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cPickle as cpl

import numpy as np

from scipy.linalg import matfuncs

from hexrd import matrixutil as mutil
from hexrd.xrd import rotations as rot
from hexrd.xrd import symmetry as symm


# =============================================================================
# %% LOCAL FUNCTIONS
# =============================================================================

# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


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


# =============================================================================
# %% TRANSFORM GRAINS.OUT DATA
# =============================================================================


gt = np.loadtxt('/Users/rachellim/Documents/Research/Dye_CHESS_Jan20/fd1-q-1_filtered/filtered_scan_0067_grains.out', ndmin=2)
rmats = rot.rotMatOfExpMap(gt[:, 3:6].T)

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
    csym.append(symm.applySym(c.reshape(3, 1), qsym, csFlag=True))
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
fig2, ax2 = plt.subplots()

pts = stereographic_zproj(sst_list, cob=cob)
ax2.plot(pts[:, 0], pts[:, 1], 'ko')

vtx = stereographic_zproj(sst_vertices.T, cob=cob)
ax2.plot(vtx[0, 0], vtx[0, 1], 'rs')
ax2.plot(vtx[1, 0], vtx[1, 1], 'gs')
ax2.plot(vtx[2, 0], vtx[2, 1], 'bs')
plt.axis('equal')

edg = stereographic_zproj(edges, cob=cob)
ax2.plot(edg[:, 0], edg[:, 1], 'k-')
