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



# ============= PREPROCESS VARIABLES ===============#

# Create recoded versions of 'Allele' categories
continuous_vars = ['iscores', 'Beta', 'SE', 'p_value', 'MAF', 'HWE-P']
al_vars = ['Allele', 'A1', 'A2']

allele_categories = ['A', 'C', 'G', 'T']

for col in al_vars:
    merged_data[col + '_recoded'] = np.where(
            merged_data[col].isin(allele_categories+[np.NaN]),
            merged_data[col], 'Mutation')

categorical_vars = [col + '_recoded' for col in al_vars]
merged_data[categorical_vars] = merged_data[categorical_vars].astype('category')

# NATURE OF 'POSITION' TO BE DECIDED LATER!



## ALTERNATIVE METHOD
#from pandas.api.types import CategoricalDtype
#
#al_vars = ['Allele', 'A1', 'A2']
#al_cat = CategoricalDtype(categories = ['A', 'C', 'G', 'T', 'Mutated'], 
#                          ordered = False)
#for col in al_vars:
#    merged_data[col + '_num'] = merged_data[col].astype(al_cat)
#    merged_data[col + '_num'] = np.where(
#            merged_data[col].isin(alleles + [np.NaN]),
#            merged_data[col], 'Mutation')
#    merged_data[col + '_num'] = merged_data[col + '_num'].cat.codes





# ============= SCALING AND ENCODING ===============#

# Convert categorical variables into binary variables
binary_vars = pd.get_dummies(merged_data[categorical_vars], columns = categorical_vars,
                             prefix = [col + '_' for col in categorical_vars])
binary_vars.index = merged_data['SNP']

# Scale continuous data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(merged_data[continuous_vars]),
                           columns = continuous_vars, index = binary_vars.index)

# Create processed dataset
processed_data = pd.concat([scaled_data, binary_vars], axis = 1)




# Scale categorical variables and binarise categorical variables
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline


#questionable_vars = ['Position']


mapper = DataFrameMapper(
  [(continuous_vars, StandardScaler()) for col in continuous_vars] +
  [(categorical_vars, LabelBinarizer()) for col in categorical_vars])

pipeline = Pipeline([("mapper", mapper),("estimator", None)])

test = pipeline.fit_transform(merged_data[continuous_vars + categorical_vars])

## Save dataframe
#results_dir = ''
#if not os.path.exists(results_dir):
#    os.mkdir(results_dir)
#    print("Directory " , dirName ,  " Created ")
#else:    
#    print("Directory " , dirName ,  " already exists")


#merged_data.to_csv('~/')