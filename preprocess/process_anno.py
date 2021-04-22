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
from complex_trait_stats.utils import ROOT, ANNOTATED_DATA


# Drop columns with too much missing data
def drop_missing_cols(data, threshold=None, return_summary=False):
    if threshold is None or threshold > 1:
        threshold = 1    # Remove columns with no data points
    na_tab = data.isnull().sum() / len(data)
    cols = na_tab[na_tab >= threshold].index.to_list()
    
    if return_summary:
        return data.drop(cols, axis=1), na_tab
    else:
        return data.drop(cols, axis=1)


# Load data
filepath = os.path.join(ROOT, "data/annovar", ANNOTATED_DATA)
anno = pd.read_csv(filepath)

# Recode "." to np.nan
anno.replace({'.':np.nan}, inplace=True)    

# Summary of missing data (No columns removed yet)
anno, na_summary = drop_missing_cols(anno, return_summary=True)
anno.drop(["cytoBand"], axis=1, inplace=True)    # Drop cytoBand column


# Split 'Gene.refGene' text data into multiple columns
anno[["Gene1", "Gene2"]] = anno["Gene.refGene"].str.split(";", expand=True)
anno.drop(["Gene.refGene"], axis=1, inplace=True)
anno.replace({None:np.nan}, inplace=True)


# Explore missing data in relation to 'Gene.refGene'
# test = anno[anno["Gene2"].isnull() == False]
# anno["Gene2"][anno["Func.refGene"]=="intronic"]

# Filter datasets by Func.refGene for exploration
utr3 = anno[anno["Func.refGene"]=="UTR3"]
ds = anno[anno["Func.refGene"]=="downstream"]
exo = anno[anno["Func.refGene"]=="exonic"]
inter = anno[anno["Func.refGene"]=="intergenic"]
intro = anno[anno["Func.refGene"]=="intronic"]
nc_intro = anno[anno["Func.refGene"]=="ncRNA_intronic"]

# print(drop_missing_cols(utr3, return_summary=True)[-1])
# print(drop_missing_cols(ds, return_summary=True)[-1])
# print(drop_missing_cols(exo, return_summary=True)[-1])
# print(drop_missing_cols(inter, return_summary=True)[-1])
# print(drop_missing_cols(intro, return_summary=True)[-1])
# print(drop_missing_cols(nc_intro, return_summary=True)[-1])

# =============================================================================
# 
# =============================================================================

filepath = os.path.join(ROOT, "data/annovar/legacy", "full_anno.hg38_multianno.csv")
anno = pd.read_csv(filepath)

# Recode "." to np.nan
anno.replace({'-':np.nan}, inplace=True) 
test, na_summary = drop_missing_cols(anno, return_summary=True)

   