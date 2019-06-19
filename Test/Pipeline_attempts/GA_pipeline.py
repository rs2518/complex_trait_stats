#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 00:23:01 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd
import os



# ============= RESULTS FILES ===============#
# Load subset of results files
directory = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/clinical_c_Block_J40-J47"


max_files = 3
counter = 0

gen_results = None
for file in os.listdir(directory):
    if file.startswith("genotyped.allWhites.") and file.endswith(".csv.gz"):
        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ",
                         compression = 'gzip')
        gen_results = pd.concat([gen_results, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files or counter >= len(os.listdir(directory)):
            break

counter = 0

imp_results = None
for file in os.listdir(directory):
    if file.startswith("imputed.allWhites.") and file.endswith(".csv.gz"):
        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ",
                         compression = 'gzip')
        imp_results = pd.concat([imp_results, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files or counter >= len(os.listdir(directory)):
            break

# Subset data
gen_results = gen_results.iloc[0:2000,]
imp_results = imp_results.iloc[0:8000,]


# ============= VARIANT INFO FILES ===============#
# Load subset of variant info files
directory2 = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/manual_downloads/variant_info"

max_files = 3
counter = 0

gen_stats = None
for file in os.listdir(directory2):
    if file.startswith("snps.genotyped.") and file.endswith(".csv"):
        df = pd.read_csv("~/" + os.path.join(directory2, file), sep = " ")
        gen_stats = pd.concat([gen_stats, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files:
            break

counter = 0

imp_stats = None
for file in os.listdir(directory2):
    if file.startswith("snps.imputed.") and file.endswith(".csv"):
        df = pd.read_csv("~/" + os.path.join(directory2, file), sep = " ")
        imp_stats = pd.concat([imp_stats, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files:
            break

# Subset data
gen_stats = gen_stats.iloc[0:1000,]
imp_stats = imp_stats.iloc[0:6000,]



# ============= PRE-PROCESSING DATA ===============#

# Rename columns, add 'Type' column, align columns and merge results data
gen_results.columns = ['SNP', 'Allele', 'Beta', 'SE', 'P_value']
imp_results.columns = ['SNP', 'Allele', 'iscores', 'Beta', 'SE', 'P_value']


unique_cols = list(imp_results.columns.difference(gen_results.columns))

gen_results['Type'] = list(['Genotyped'] * gen_results.shape[0])
imp_results['Type'] = list(['Imputed'] * imp_results.shape[0])
imp_results = imp_results[list(gen_results.columns) + unique_cols]

res_data = pd.concat([gen_results,imp_results], axis = 0, sort = False)


# Add 'Type' to dataframes, align columns and merge stats data
unique_cols = list(imp_stats.columns.difference(gen_stats.columns))

gen_stats['Type'] = list(['Genotyped'] * gen_stats.shape[0])
imp_stats['Type'] = list(['Imputed'] * imp_stats.shape[0])
imp_stats = imp_stats[list(gen_stats.columns) + unique_cols]

stats_data = pd.concat([gen_stats,imp_stats], axis = 0, sort = False)


# Merge variant info and results data
merged_data = pd.merge(res_data, stats_data, right_on = 'SNP',
                     left_on = 'SNP', how = 'outer')


# Find row indices for iscores that agree or disagree. Check for any overlap
# Agree if both entries are missing or contain same value
iscores_agree = (merged_data['iscores'].isna() & merged_data['iscore'].isna()) | (merged_data['iscores'] == merged_data['iscore'])
type_agree = (merged_data['Type_x'].isna() & merged_data['Type_y'].isna()) | (merged_data['Type_x'] == merged_data['Type_y'])


############
## (OPTIONAL) Check indices that disagree and check that there is no overlap
#iscores_disagree = ((merged_data['iscores'].isna() == False) | (merged_data['iscore'].isna() == False)) | (merged_data['iscores'] != merged_data['iscore'])
#
#print(any(index in list(iscores_disagree) for index in list(iscores_agree)))
#
#
#
# (OPTIONAL) Check Type for any overlap. Should be disjoint
#type_disagree = ((merged_data['Type_x'].isna() == False) | (merged_data['Type_y'].isna() == False)) & (merged_data['Type_x'] != merged_data['Type_y'])
#            
#print(any(index in list(type_disagree) for index in list(type_agree)))
############


# Recode false missing data in 'iscore' and 'Type' and drop redundant columns
indices_iscore = list(merged_data[(iscores_agree == False) & (merged_data['iscore'].isnull())].index)
for index in indices_iscore:
    merged_data.loc[index, 'iscore'] = merged_data.loc[index, 'iscores']

indices_type = list(merged_data[(type_agree == False) & (merged_data['Type_x'].isnull())].index)
for index in indices_type:
    merged_data.loc[index, 'Type_x'] = merged_data.loc[index, 'Type_y']


merged_data.rename({'Type_x':'Type'}, axis = 1, inplace = True)
merged_data.drop(['Type_y','iscores'], axis = 1, inplace = True)

# Scale continuous variables


# Create column of labels (association to complex trait)
signif_thres = 5 * 10**-5
merged_data['Association?'] = np.where(merged_data['P_value'] <= signif_thres, 1, 0)


## Save dataframe
#results_dir = ''
#if not os.path.exists(results_dir):
#    os.mkdir(results_dir)
#    print("Directory " , dirName ,  " Created ")
#else:    
#    print("Directory " , dirName ,  " already exists")


#merged_data.to_csv('~/')