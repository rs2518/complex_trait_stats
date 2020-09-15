# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

import pandas as pd

from complex_trait_stats.utils import ROOT


# # Load exome array data
# # ---------------------

# path = os.path.join(ROOT, "data")
# filepath = os.path.join(path, "giant/2018_exome_array",
#                         "height_All_add_SV.txt")
# data = pd.read_csv(filepath, sep="\t")

                
# # Save sample 
# # -----------
# # Save random sample of dataframe

# # Sampled data
# sample = data.sample(n = 500, random_state = 1010)
# sample.to_csv(os.path.join(directory, "giant_exome.txt"))

# print('Data saved!')

# # Save SNP list
# snps = data["SNPNAME"].sample(n=400, random_state=1010)
# snps.to_csv(os.path.join(path, "annovar", "giant_exome_snps.txt"),
#             sep=" ", header=None, index=None)

# #### NOTE: DtypeWarning may appear because low_memory option was not properly
# # deprecated



# Load Meta-analysis data
# -----------------------

path = os.path.join(ROOT, "data")
filepath = os.path.join(path, "giant/2018_ukbb_meta",
                        "Meta-analysis_Wood_et_al+UKBiobank_2018.txt")
data = pd.read_csv(filepath, sep="\t")


# Save sample 
# -----------

# Move SNP name to first column and save random sample of dataframe
cols = data.columns.to_list()
cols.insert(0, cols.pop(cols.index("SNP")))
data = data[cols]

# sample = data.sample(n = 500, random_state = 1010)
# sample.to_csv(os.path.join(path, "annovar", "giant_meta.txt"),
#               sep=" ", header=None, index=None)

# Save SNP list
snps = data["SNP"].sample(n=400, random_state=1010)
snps.to_csv(os.path.join(path, "annovar", "giant_meta_snps.txt"),
            sep=" ", header=None, index=None)

#### NOTE: DtypeWarning may appear because low_memory option was not properly
# deprecated

print('Data saved!')