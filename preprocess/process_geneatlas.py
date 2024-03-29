import os
import gzip

import numpy as np
import pandas as pd



# Load data
# ---------
# Randomly select row indices then read randomly selected rows of data

root_dir = "/rds/general/project/medbio-berlanga-group/live/projects/ml_trait_prediction/scripts/complex_trait_stats"
directory = os.path.join(root_dir, "data/geneatlas")
trait = "50-0.0"
path = os.path.join(directory, "results", trait)


# Set sample size (for each file) and max number of files to read
# max_files = 4
sample = 1000
r = np.random.RandomState(0)     # Set seed


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
counter = 0
merged_data = None
for file in results_files:
    
    filepath = os.path.join(path, file)
    
    # Define rows to be skipped
    if sample is not None:
        with gzip.open(filepath, 'rb') as f:
            nrows = sum(1 for line in f)
            f.close()
        skip_rows = r.choice(np.arange(1, nrows), size=nrows-sample-1,
                             replace = False)
    
    # Find chromosome number and load subset of data
    chr_no = file[file.find('.chr')+4:file.find(ext)]
    imp_results = pd.read_csv(filepath,
                     sep = ' ',
                     compression = 'gzip',
                     skiprows = skip_rows)
    imp_results["Chromosome"] = ["chr" + chr_no] * len(imp_results)

    # Load variant info files and append with corresponding results file
    for file in variant_files:
        
        # Check if chromosome numbers match and merge files
        chr_no2 = file[file.find('.chr')+4:file.find(ext)]
        if chr_no == chr_no2:
            
            imp_stats = pd.read_csv(os.path.join(directory, file),
                             sep = ' ')
            imp_stats["Chromosome"] = ["chr" + chr_no2] * len(imp_stats)
            
            df = pd.merge(imp_results, imp_stats,
                          right_on = ["SNP", "Chromosome"],
                          left_on = ["SNP", "Chromosome"],
                          how = 'inner')
            
            merged_data = pd.concat([merged_data, df], axis = 0)
            break
    
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
# renamed_cols = {'ALLELE' : 'Allele',
#                 'NBETA-50-0.0' : 'Beta',
#                 'NSE-50-0.0' : 'SE',
#                 'PV-50-0.0' : 'p_value',
#                 'HWE-P' : 'HWE_P',
#                 'Position' : 'start_position'}
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

# # Create 'end_position' column and drop redundant columns
# merged_data['tmp'] = \
#     [len(a)-len(b) for a, b in zip(merged_data['A1'], merged_data['A2'])]
# merged_data['end_position'] = \
#     merged_data['start_position'] + merged_data['tmp']
# merged_data.drop(['iscore', 'Allele', 'tmp'], axis = 1, inplace = True)


# Reorder columns
main = ["Chromosome", "Position", "A1", "A2"]
# main = ["Chromosome", "start_position", "end_position", "A1", "A2"]
cols = merged_data.columns.to_list()
cols.append(cols.pop(cols.index("p_value")))
cols = main + [col for col in cols if col not in main]
merged_data = merged_data[cols]


                
# Save dataframe
# --------------
# Save dataframes or random sample of dataframes in csv format
save_path = os.path.join(root_dir, "data")

# # Sampled data
# merged_data.sample(n=500, random_state=1010).to_csv(
#     os.path.join(save_path, "sample500_geneatlas_data_22chr_1000n.csv"))


# Full data
merged_data.to_csv(os.path.join(save_path, "geneatlas_data_22chr_1000n.csv"))
