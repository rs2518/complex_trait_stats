# =============================================================================
# Old GeneATLAS pipeline
# =============================================================================

# ============= LOAD LIBRARIES =============== #

import numpy as np
import pandas as pd
import os
import gzip

from sklearn.preprocessing import StandardScaler



# ============= LOAD AND MERGE DATA FILES =============== #


##### LOADS RANDOM SAMPLE OF SNPS

# Set directories and load raw data
os.chdir(os.path.expanduser('~'))
directory = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/50-0.0'
directory2 = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/manual_downloads/variant_info'


# Set iterables and sample size (for each file)
#max_files = 5
sample = 1000
counter = 0
merged_data = None


# Create list of results and variant info files
results_files = [file for file in sorted(os.listdir(directory))
 if (file.startswith('imputed.allWhites.')
 and file.endswith('.csv.gz')
 and file.find('.chr') != -1)
 and file[file.find('.chr')+4:file.find('.csv.gz')].isdigit()]

variant_files = [file for file in sorted(os.listdir(directory2))
 if (file.startswith('snps.imputed.')
 and file.endswith('.csv')
 and file.find('.chr') != -1)
 and file[file.find('.chr')+4:file.find('.csv')].isdigit()]


# Load and append results files
for file in results_files:
    
    # Define rows to be skipped
    filepath = os.path.join(directory, file)
    with gzip.open(filepath, 'rb') as f:
        nrows = sum(1 for line in f)
        f.close()
    np.random.seed(0)     # Set seed
    skip_rows = np.random.choice(np.arange(1, nrows),
                                 size = nrows - sample - 1,
                                 replace = False)
    
    # Find chromosome number and load subset of data
    chr_no = int(file[file.find('.chr')+4:file.find('.csv.gz')])
    imp_results = pd.read_csv('~/' + filepath,
                     sep = ' ',
                     compression = 'gzip',
                     skiprows = skip_rows)
    imp_results['Chr_no'] = chr_no * np.ones(
            len(imp_results), dtype = np.int8)


    # Load variant info files and append with corresponding results file
    for file2 in variant_files:
        
        # Check if chromosome numbers match and merge files
        chr_no2 = int(file2[file2.find('.chr')+4:file2.find('.csv')])
        if chr_no == chr_no2:
            
            imp_stats = pd.read_csv('~/' + os.path.join(directory2, file2),
                             sep = ' ')
            imp_stats['Chr_no'] = chr_no2 * np.ones(
                    len(imp_stats), dtype = np.int8)
            
            df = pd.merge(imp_results, imp_stats,
                          right_on = ['SNP', 'Chr_no'],
                          left_on = ['SNP', 'Chr_no'],
                          how = 'inner')
            
            merged_data = pd.concat([merged_data, df], axis = 0)
    
    # Increment counter and break if loops exceeds max number of files
    counter = counter + 1
    try:
        if counter >= max_files or counter >= len(
                results_files) or counter >= len(
                        variant_files):
            break
    except NameError:
        if counter >= len(
                results_files) or counter >= len(
                        variant_files):
            break

# Re-index data to remove repeated indices
merged_data.index = np.arange(len(merged_data))



# ============= MERGE DATASETS =============== #

# Rename columns
renamed_cols = {'ALLELE' : 'Allele',
                'NBETA-50-0.0' : 'Beta',
                'NSE-50-0.0' : 'SE',
                'PV-50-0.0' : 'p_value',
                'HWE-P' : 'HWE_P'}
merged_data.rename(columns = renamed_cols, inplace = True)


# Recode false missing data in 'iscore'
iscores_agree = (merged_data['iscores'].isna() & merged_data[
        'iscore'].isna()) | (merged_data['iscores'] == merged_data['iscore'])

indices_iscore = list(merged_data[
        (iscores_agree == False) & (merged_data['iscores'].isnull())].index)
for index in indices_iscore:
    merged_data.loc[index, 'iscores'] = merged_data.loc[index, 'iscore']


# Drop redundant columns
merged_data.drop(['iscore', 'Allele'], axis = 1, inplace = True)



# ============= RECODE VARIABLES =============== #

# Convert chromosome number to nominal
merged_data['Chr_no'] = pd.Categorical(merged_data['Chr_no'],
               categories = list(range(1,23)), ordered = False)


# Convert allele variable to nominal
al_vars = ['A1', 'A2']
alleles = ['A', 'C', 'G', 'T']
new_category = 'Other'
al_categories = alleles + [new_category]
for col in al_vars:
    merged_data[col + '_v2'] = np.where(
            merged_data[col].isin(alleles + [np.NaN]),
            merged_data[col], new_category)
    merged_data[col + '_v2'] = merged_data[col + '_v2'].astype('category')


# Define continuous and categorical variables to be pre-processed.
# 'exceptions' will not be pre-processed
exceptions = ['Chr_no', 'Position', 'p_value']

continuous_vars = [col for col in merged_data.select_dtypes(
        exclude=['object', 'category']).columns
    if col not in exceptions]

categorical_vars = [col for col in merged_data.select_dtypes(
        include=['category']).columns
    if col not in exceptions]


# NOTE: 'POSITION' TO BE CONSIDERED AS A NUMERICAL ID!



# ============= SCALING AND ENCODING =============== #

# Scale categorical variables and binarise categorical variables
scaler = StandardScaler()
scaled_data = pd.DataFrame(
        scaler.fit_transform(merged_data[continuous_vars]),
        columns = [col + '_scaled' for col in continuous_vars])

dummy_data = pd.get_dummies(merged_data[categorical_vars],
                           columns = categorical_vars,
                           prefix = [col + '_' for col in categorical_vars])

processed_data = pd.concat([merged_data[['SNP', 'p_value']],
                            scaled_data,
                            dummy_data], axis = 1)


# Reorder columns and log transform p_values
processed_cols = processed_data.columns.to_list()
ind = processed_cols.index('p_value')
processed_cols = processed_cols[0:ind] + processed_cols[ind+1:] + [
        processed_cols[ind]]

merged_cols = merged_data.columns.to_list()
ind = merged_cols.index('p_value')
merged_cols = merged_cols[0:ind] + merged_cols[ind+1:] + [
        merged_cols[ind]]

processed_data = processed_data[processed_cols]
processed_data['log_p_val'] = -np.log10(processed_data['p_value'])
merged_data = merged_data[merged_cols]
merged_data['log_p_val'] = -np.log10(merged_data['p_value'])



# ============= SAVE DATA =============== #

## Check if directory exists. If not, create a new directory
#if not os.path.exists('Processed'):
#    os.mkdir('Processed')
#    print('Created './Processed/' directory')
#    processed_data.to_pickle(os.path.join('./Processed', 
#                                       'processed_data.pkl'))
#    merged_data.to_pickle(os.path.join('./Processed', 
#                                       'integrated_data.pkl'))
#else:    
#    print(''./Processed/' directory already exists')
#    processed_data.to_pickle(os.path.join('./Processed', 
#                                       'processed_data.pkl'))
#    merged_data.to_pickle(os.path.join('./Processed', 
#                                       'integrated_data.pkl'))



# Locally save sample
os.chdir('Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/')
seed = 1
sample_processed = processed_data.sample(n = 500, random_state = seed)
sample_merged = merged_data.sample(n = 500, random_state = seed)

if not os.path.exists('Processed'):
    os.mkdir('Processed')
    print('Created \'./Processed/\' directory')
    sample_processed.to_pickle(os.path.join('./Processed', 
                                       'processed_data_allchr.pkl'))
    sample_merged.to_pickle(os.path.join('./Processed', 
                                       'integrated_data_allchr.pkl'))
else:    
    print('\'./Processed/\' directory already exists')
    sample_processed.to_pickle(os.path.join('./Processed', 
                                       'processed_data_allchr.pkl'))
    sample_merged.to_pickle(os.path.join('./Processed', 
                                       'integrated_data_allchr.pkl'))