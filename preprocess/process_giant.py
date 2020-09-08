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


# Load data
# ---------
# Load GIANT height data

path = os.path.join(ROOT, "data/giant")
filepath = os.path.join(path, "height_All_add_SV.txt")
data = pd.read_csv(filepath, sep="\t")

                
# # Save sample 
# # -----------
# # Save random sample of dataframe

# # Sampled data
# sample = data.sample(n = 500, random_state = 1010)
# sample.to_csv(os.path.join(directory, "giant.txt"))

# print('Data saved!')

# Save SNP list
snps = data["SNPNAME"].sample(n=400, random_state=1010)
snps.to_csv(os.path.join(path, "giant_snps.txt"),
            sep=" ", header=None, index=None)