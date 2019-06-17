#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:11:21 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



# ============= LOAD DATA FILES =============== #

##### Results files
# Load subset of results files
directory = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/Data/50-0.0"


max_files = 3
counter = 0


imp_results = None
for file in sorted(os.listdir(directory)):
    if file.startswith("imputed.allWhites.") and file.endswith(".csv.gz") and file.find('.chr') != -1:
        df = pd.read_csv("~/" + os.path.join(directory, file), sep = " ",
                         compression = 'gzip')
        
        # Add column for chr no.
        chromosome_no = int(file[file.find('.chr')+4:file.find('.csv.gz')])
        df['Chr_no'] = chromosome_no * np.ones(len(df))
        imp_results = pd.concat([imp_results, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files or counter >= len(os.listdir(directory)):
            break

# Subset data
imp_results = imp_results.sample(n = 8000, random_state = 1)


##### Variant info files
# Load subset of variant info files
directory2 = "Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/Data/manual_downloads/variant_info"

max_files = 3
counter = 0


imp_stats = None
for file in sorted(os.listdir(directory2)):
    if file.startswith("snps.imputed.") and file.endswith(".csv") and file.find('.chr') != -1:
        df = pd.read_csv("~/" + os.path.join(directory2, file), sep = " ")

        # Add column for chr no.
        chromosome_no = int(file[file.find('.chr')+4:file.find('.csv')])
        df['Chr_no'] = chromosome_no * np.ones(len(df))
        imp_stats = pd.concat([imp_stats, df], axis = 0)
        
        counter = counter + 1
        if counter >= max_files:
            break

# Subset data
imp_stats = imp_stats.sample(n = 12000, random_state = 1)



# ============= MERGE DATASETS ===============#

# Rename columns and merge data
imp_results.columns = ['SNP', 'Allele', 'iscores', 'Beta', 'SE',
                       'p_value', 'Chr_no']
merged_data = pd.merge(imp_results, imp_stats, right_on = ['SNP', 'Chr_no'],
                     left_on = ['SNP', 'Chr_no'], how = 'outer')


# View distribution of labels (i.e. 'p_value')
sns.distplot(merged_data['p_value'].dropna())
plt.show()     # Relatively even distribution with a spike around p_value = 0


# Find row indices for iscores that agree or disagree. Check for any overlap
iscores_agree = (merged_data['iscores'].isna() & merged_data['iscore'].isna()) | (merged_data['iscores'] == merged_data['iscore'])
# Agree if both entries are missing or contain same value


# Recode false missing data in 'iscore' and 'Type' and drop redundant columns
indices_iscore = list(merged_data[(iscores_agree == False) & (merged_data['iscores'].isnull())].index)
for index in indices_iscore:
    merged_data.loc[index, 'iscores'] = merged_data.loc[index, 'iscore']

merged_data.drop(['iscore'], axis = 1, inplace = True)



# ============= RECODE VARIABLES ===============#

# Convert chromosome number to ordinal
merged_data['Chr_no'] = pd.Categorical(merged_data['Chr_no'],
               categories = list(range(1,23)), ordered = True)


# Convert allele variable to nominal
al_vars = ['Allele', 'A1', 'A2']

alleles = ['A', 'C', 'G', 'T']
new_category = 'Mutation'
al_categories = alleles + [new_category]
for col in al_vars:
    merged_data[col + '_v2'] = np.where(
            merged_data[col].isin(alleles + [np.NaN]),
            merged_data[col], new_category)
    merged_data[col + '_v2'] = pd.Categorical(merged_data[col], 
               categories = al_categories, ordered = False)


# Define continuous and categorical variables
continuous_vars = ['iscores', 'Beta', 'SE', 'p_value', 'MAF', 'HWE-P']
categorical_vars = [col + '_v2' for col in al_vars] + ['Chr_no']

# NATURE OF 'POSITION' TO BE DECIDED LATER!



# ============= SCALING AND ENCODING ===============#

# Scale continuous data and binarise categorical variables 
from sklearn.preprocessing import StandardScaler

binary_df = pd.get_dummies(merged_data[categorical_vars],
                           columns = categorical_vars,
                           prefix = [col + '_' for col in categorical_vars],
                           dummy_na = True)
#binary_df.index = merged_data['SNP']
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(merged_data[continuous_vars]),
                           columns = continuous_vars)

# Create processed dataset
processed_data = pd.concat([scaled_df, binary_df], axis = 1)
processed_data.index = merged_data['SNP']


# pd.get_dummies and LabelBinarizer CANNOT handle missing data



