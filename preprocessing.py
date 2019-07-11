#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:11:21 2019

@author: raphaelsinclair
"""


# ============= LOAD DATA FILES =============== #

# Set directories and load raw data
import numpy as np
import pandas as pd
import os

#hpc_path = '/rdsgpfs/general/project/medbio-berlanga-group/live/projects/ml_trait_prediction/Data'
#os.chdir(hpc_path)
#directory = 'Raw'

os.chdir(os.path.expanduser('~'))
directory = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/50-0.0'
directory2 = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/manual_downloads/variant_info'

max_files = 2


# Loading results files
counter = 0
imp_results = None
for file in sorted(os.listdir(directory)):
    if file.startswith('imputed.allWhites.') and file.endswith('.csv.gz') and file.find('.chr') != -1:
        df = pd.read_csv('~/' + os.path.join(directory, file), sep = ' ',
                         compression = 'gzip')
        
        # Add column for chr no.
        chromosome_no = int(file[file.find('.chr')+4:file.find('.csv.gz')])
        df['Chr_no'] = chromosome_no * np.ones(len(df), dtype = np.int8)
        imp_results = pd.concat([imp_results, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files or counter >= len(os.listdir(directory)):
            break


# Loading variant info files
counter = 0
imp_stats = None
for file in sorted(os.listdir(directory2)):
    if file.startswith('snps.imputed.') and file.endswith('.csv') and file.find('.chr') != -1:
        df = pd.read_csv('~/' + os.path.join(directory2, file), sep = ' ')

        # Add column for chr no.
        chromosome_no = int(file[file.find('.chr')+4:file.find('.csv')])
        df['Chr_no'] = chromosome_no * np.ones(len(df), dtype = np.int8)
        imp_stats = pd.concat([imp_stats, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files:
            break



# ============= MERGE DATASETS =============== #

# Rename columns and merge data
imp_results.columns = ['SNP', 'Allele', 'iscores', 'Beta', 'SE',
                       'p_value', 'Chr_no']
imp_stats.columns = ['SNP', 'Position', 'A1', 'A2', 'MAF', 'HWE_P',
                     'iscore', 'Chr_no']

merged_data = pd.merge(imp_results, imp_stats, right_on = ['SNP', 'Chr_no'],
                     left_on = ['SNP', 'Chr_no'], how = 'outer')


# Find row indices for iscores that agree or disagree. Check for any overlap
iscores_agree = (merged_data['iscores'].isna() & merged_data['iscore'].isna()) | (merged_data['iscores'] == merged_data['iscore'])
# Agree if both entries are missing or contain same value


# Recode false missing data in 'iscore' and drop redundant column
indices_iscore = list(merged_data[(iscores_agree == False) & (merged_data['iscores'].isnull())].index)
for index in indices_iscore:
    merged_data.loc[index, 'iscores'] = merged_data.loc[index, 'iscore']

merged_data.drop(['iscore'], axis = 1, inplace = True)



# ============= RECODE VARIABLES =============== #

# Convert chromosome number to nominal
merged_data['Chr_no'] = pd.Categorical(merged_data['Chr_no'],
               categories = list(range(1,23)), ordered = False)


# Convert allele variable to nominal
al_vars = ['Allele', 'A1', 'A2']

alleles = ['A', 'C', 'G', 'T']
new_category = 'Other'
al_categories = alleles + [new_category]
for col in al_vars:
    merged_data[col + '_v2'] = np.where(
            merged_data[col].isin(alleles + [np.NaN]),
            merged_data[col], new_category)
    merged_data[col + '_v2'] = merged_data[col + '_v2'].astype('category')


# Define continuous and categorical variables
num_exceptions = ['Position']
continuous_vars = [col for col in merged_data.select_dtypes(
        exclude=['object', 'category']).columns
    if col not in num_exceptions]
categorical_vars = merged_data.select_dtypes(include=['category']).columns.to_list()


# 'POSITION' TO BE CONSIDERED AS A NUMERICAL ID!



# ============= SCALING AND ENCODING =============== #

# Scale categorical variables and binarise categorical variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(merged_data[continuous_vars]),
                           columns = [col + '_scaled' for col in continuous_vars])

dummy_data = pd.get_dummies(merged_data[categorical_vars],
                           columns = categorical_vars,
                           prefix = [col + '_' for col in categorical_vars])

processed_data = pd.concat([merged_data, scaled_data, dummy_data], axis = 1)


# Reorder columns and log transform p_values
processed_cols = processed_data.columns.to_list()
processed_cols = processed_cols[0:5] + processed_cols[6:] + [processed_cols[5]]
merged_cols = merged_data.columns.to_list()
merged_cols = merged_cols[0:5] + merged_cols[6:] + [merged_cols[5]]

processed_data = processed_data[processed_cols]
processed_data['log_p_val'] = np.log10(processed_data['p_value'])
merged_data = merged_data[merged_cols]
merged_data['log_p_val'] = -np.log10(merged_data['p_value'])



# ============= SAVE DATA =============== #

## Check if directory exists. If not, create a new directory
#if not os.path.exists('Processed'):
#    os.mkdir('Processed')
#    print('Created \'./Processed/\' directory')
#    processed_data.to_csv(os.path.join('./Processed', 
#                                       'processed_data.csv'))
#    merged_data.to_csv(os.path.join('./Processed', 
#                                       'integrated_data.csv'))
#else:    
#    print('\'./Processed/\' directory already exists')
#    processed_data.to_csv(os.path.join('./Processed', 
#                                       'processed_data.csv'))
#    merged_data.to_csv(os.path.join('./Processed', 
#                                       'integrated_data.csv'))



# Locally save sample
os.chdir('Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/')
sample_processed = processed_data.sample(n = 500, random_state = 1)
sample_merged = merged_data.sample(n = 500, random_state = 1)


if not os.path.exists('Processed'):
    os.mkdir('Processed')
    print('Created \'./Processed/\' directory')
    sample_processed.to_pickle(os.path.join('./Processed', 
                                       'processed_data_sample.pkl'))
    sample_merged.to_pickle(os.path.join('./Processed', 
                                       'integrated_data_sample.pkl'))
else:    
    print('\'./Processed/\' directory already exists')
    sample_processed.to_pickle(os.path.join('./Processed', 
                                       'processed_data_sample.pkl'))
    sample_merged.to_pickle(os.path.join('./Processed', 
                                       'integrated_data_sample.pkl'))