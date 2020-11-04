# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

import numpy as np
import pandas as pd

# from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import ROOT


# filepath = os.path.join(ROOT, "data/annovar", "myanno.hg19_multianno.csv")
# anno = pd.read_csv(filepath)
filepath = os.path.join(ROOT, "data/annovar", "giantanno.hg19_multianno.csv")
anno = pd.read_csv(filepath)

anno.replace({'.':np.nan}, inplace=True)    # Recode "." to np.nan


# Drop columns with too much missing data
threshold = 30
na_tab = anno.isnull().sum() * 100 / len(anno)
cols = na_tab[na_tab >= threshold].index.to_list()

# anno.drop(cols, axis=1, inplace=True)


# Split 'Gene.refGene' text data into multiple columns
anno[["refGene1", "refGene2"]] = \
    anno["Gene.refGene"].str.split(";", expand=True)
anno.drop(["Gene.refGene"], axis=1, inplace=True)
anno.replace({None:np.nan}, inplace=True)


# Explore missing data in relation to 'Gene.refGene'
test = anno[not anno["refGene2"].isnull()]
test2 = anno[anno["refGene2"].isnull() == False]