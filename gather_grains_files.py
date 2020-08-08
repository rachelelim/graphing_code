#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:32:04 2020

@author: rachellim
"""

import numpy as np
import os
import shutil

directory = '/nfs/chess/user/bernier2/Working/dye-869-1'
grains_path = 'results_fd1-a_%04d*/grains.out'


for i in range(4,87):
    filepath = os.path.join(directory,grains_path%i)
    new_filepath = '/nfs/chess/user/relim/dye-869-1/fd1-a-1/scan_%04d_grains.out'
    shutil.copy(filepath,new_filepath)