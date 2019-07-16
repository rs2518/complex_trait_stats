#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:38:29 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd
import os

# ========== INTEGRATING DATA ========== #

# Load and concatenate all genotyped variant info files in directory
directory = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/manual_downloads/variant_info"

genotyped_stats = None
for genotyped_file in os.listdir(directory):
    if genotyped_file.startswith("snps.genotyped") and genotyped_file.endswith(".csv"):
        df = pd.read_csv("~/" + os.path.join(directory, genotyped_file), sep = " ")
        genotyped_stats = pd.concat([genotyped_stats, df], axis = 0)


# Load and concatenate all imputed variant info files in directory
imputed_stats = None
for imputed_file in os.listdir(directory):
    if imputed_file.startswith("snps.imputed") and imputed_file.endswith(".csv"):
        df = pd.read_csv("~/" + os.path.join(directory, imputed_file), sep = " ")
        imputed_stats = pd.concat([imputed_stats, df], axis = 0)




##### SUBSET DATA #####
# NOTE: Comment out for HPC
genotyped_stats = genotyped_stats.iloc[0:1000,]
imputed_stats = imputed_stats.iloc[0:6000,]
#######################



# Add 'Type' to dataframes, align columns and merge data
#unique_cols = []
#for column in list(imputed_stats.columns):
#    if column not in list(genotyped_stats.columns):
#        unique_cols.append(column)

unique_cols = list(imputed_stats.columns.difference(genotyped_stats.columns))

genotyped_stats['Type'] = list(['Genotyped'] * genotyped_stats.shape[0])
imputed_stats['Type'] = list(['Imputed'] * imputed_stats.shape[0])
imputed_stats = imputed_stats[list(genotyped_stats.columns) + unique_cols]

var_info_data = pd.concat([genotyped_stats,imputed_stats], axis = 0, sort = False)

