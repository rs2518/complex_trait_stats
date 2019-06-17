#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:58 2019

@author: raphaelsinclair
"""

import pandas as pd

# ============= PROCESS TRAIT TABLE ===============#

# Load Trait Table and extract trait keys
path = '~/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/'
trait_table = pd.read_excel(path+'Traits_Table_GeneATLAS.xlsx', sheet_name = 'TraitTable', skiprows = 1)
keys = list(trait_table.iloc[:,0])


# Create array of arguments for bash
# Size = maximum number of traits loaded simultaneously in bash script
size = 10
index = 0
trait_array = []
                      
while index <= (len(keys) // size):
    if index == (len(keys) // size):
        trait_array.append(["\'" + key + "\'" for key in keys[(index * 10):(index * 10) + len(keys) % size]])
    else:
        trait_array.append(["\'" + key + "\'" for key in keys[(index * 10):((index + 1) * 10)]])
    index = index + 1
