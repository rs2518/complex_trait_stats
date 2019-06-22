#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:43:46 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd
import os

# ========== INTEGRATING DATA ========== #

# Load and concatenate all genotyped variant info files in directory
directory = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/clinical_c_Block_J40-J47"



###### Load ALL files
#gen_results = None
#imp_results = None
#
#for file in os.listdir(directory):
#    if file.startswith("genotyped.allWhites.") and file.endswith(".csv.gz"):
#        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ", compression = 'gzip')
#        gen_results = pd.concat([gen_results, df], axis = 0)
#
#    elif file.startswith("imputed.allWhites.") and file.endswith(".csv.gz"):
#        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ", compression = 'gzip')
#        imp_results = pd.concat([imp_results, df], axis = 0)



#########################################

##### Load small selection of files #####
# Comment out when loading all files
max_files = 3

counter = 0
gen_results = None
for file in os.listdir(directory):
    if file.startswith("genotyped.allWhites.") and file.endswith(".csv.gz"):
        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ", compression = 'gzip')
        gen_results = pd.concat([gen_results, df], axis = 0)
        counter = counter + 1
        if counter >= max_files:
            break

counter = 0
imp_results = None
for file in os.listdir(directory):
    if file.startswith("imputed.allWhites.") and file.endswith(".csv.gz"):
        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ", compression = 'gzip')
        imp_results = pd.concat([imp_results, df], axis = 0)
        counter = counter + 1
        if counter >= max_files:
            break

# Subset data
gen_results = gen_results.iloc[0:2000,]
imp_results = imp_results.iloc[0:8000,]


#########################################


# Rename and reorder columns
gen_results.columns = ['SNP', 'Allele', 'Beta', 'SE', 'P_value']
imp_results.columns = ['SNP', 'Allele', 'iscores', 'Beta', 'SE', 'P_value']


# Add 'Type' to dataframes, align columns and merge data
unique_cols = list(imp_results.columns.difference(gen_results.columns))

gen_results['Type'] = list(['Genotyped'] * gen_results.shape[0])
imp_results['Type'] = list(['Imputed'] * imp_results.shape[0])
imp_results = imp_results[list(gen_results.columns) + unique_cols]

res_data = pd.concat([gen_results,imp_results], axis = 0, sort = False)

