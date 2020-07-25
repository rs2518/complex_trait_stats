import os
import gzip

import numpy as np
import pandas as pd



# Load data
# ---------
# Randomly select row indices then read randomly selected rows of data

directory = "Desktop/Term 3 MSc Project/complex_trait_stats/data"
trait = "50-0.0"
path = os.path.join(directory, "results/", trait)


# Set iterables and sample size (for each file)
max_files = 5
sample = 1000
counter = 0
merged_data = None

# Create list of results and variant info files
ext = ".csv.gz"
results_files = [file for file in sorted(os.listdir(path))
 if (file.startswith('imputed.allWhites.')
 and file.endswith(ext)
 and file.find('.chr') != -1)
 and file[file.find('.chr')+4:file.find(ext)].isdigit()]

variant_files = [file for file in sorted(os.listdir(directory))
 if (file.startswith('snps.imputed.')
 and file.endswith(ext)
 and file.find('.chr') != -1)
 and file[file.find('.chr')+4:file.find(ext)].isdigit()]


# Load and append results files
for file in results_files:
    
    # Define rows to be skipped
    filepath = os.path.join(path, file)
    with gzip.open(filepath, 'rb') as f:
        nrows = sum(1 for line in f)
        f.close()
    np.random.seed(0)     # Set seed
    skip_rows = np.random.choice(np.arange(1, nrows),
                                 size = nrows - sample - 1,
                                 replace = False)
    
    # Find chromosome number and load subset of data
    chr_no = int(file[file.find('.chr')+4:file.find(ext)])
    imp_results = pd.read_csv('~/' + filepath,
                     sep = ' ',
                     compression = 'gzip',
                     skiprows = skip_rows)
    imp_results['Chr_no'] = chr_no * np.ones(
            len(imp_results), dtype = np.int8)


    # Load variant info files and append with corresponding results file
    for file in variant_files:
        
        # Check if chromosome numbers match and merge files
        chr_no2 = int(file[file.find('.chr')+4:file.find(ext)])
        if chr_no == chr_no2:
            
            imp_stats = pd.read_csv('~/' + os.path.join(directory, file),
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




# Characterise merged dataframe
# -----------------------------
# Some of the missing data may appearing in the dataframe is not actually
# missing but instead appears in a different column from failure to merge

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

# Log-transformed p-values
merged_data['log_p_val'] = -np.log10(merged_data['p_value'])



# Save dataframe
# -----------------------------
# Save random sample of merged dataframe

# CSV format
sample_merged = merged_data.sample(n = 500, random_state = 1010)
sample_merged.to_csv(os.path.join(directory, "snp_data.csv"))
print('Data saved!')