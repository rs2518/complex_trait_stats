# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import numpy as np
import pandas as pd

from complex_trait_stats.utils import ROOT



# ===== Investigate datasets ===== #
# The following files are the different types of files found for each trait in GeneATLAS
# NOTE: '*_stats' and 'hla freq' files must be downloaded manually

# path = "~/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/Data/50-0.0/"
path = ROOT+"/data/results/50-0.0/"

gen_res = pd.read_csv(path+"genotyped.allWhites.50-0.0.chr22.csv.gz",
                      sep=" ", compression='gzip')
gen_nr = pd.read_csv(path+"genotyped.normRank.allWhites.50-0.0.chr22.csv.gz",
                     sep = " ", compression = 'gzip')

imp_res = pd.read_csv(path+"imputed.allWhites.50-0.0.chr22.csv.gz",
                      sep=" ", compression='gzip')
imp_nr = pd.read_csv(path+"imputed.normRank.allWhites.50-0.0.chr22.csv.gz",
                     sep = " ", compression = 'gzip')


# gen_stats = pd.read_csv(ROOT+"/data/snps.genotyped.chr22.csv.gz",
#                         sep=" ", compression='gzip')
imp_stats = pd.read_csv(ROOT+"/data/snps.imputed.chr22.csv.gz",
                        sep=" ", compression='gzip')

hla = pd.read_csv(path+"hla.50-0.0.csv.gz", sep = " ", compression = 'gzip')
hla_omnibus = pd.read_csv(path+"hla.omnibus.50-0.0.csv.gz",
                          sep = " ", compression = 'gzip')

hla_freq = pd.read_csv(path+"hla.freq", sep = " ")


