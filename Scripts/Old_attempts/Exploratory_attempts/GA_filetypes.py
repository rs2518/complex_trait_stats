#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:03:16 2019

@author: raphaelsinclair
"""

import pandas as pd
import numpy as np



# ===== Investigate datasets ===== #
# The following files are the different types of files found for each trait in GeneATLAS
# NOTE: '*_stats' and 'hla freq' files must be downloaded manually

path = "~/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/Data/50-0.0/"

genotyped_res = pd.read_csv(path+"genotyped.allWhites.50-0.0.chr22.csv.gz", sep = " ", compression = 'gzip')
imputed_res = pd.read_csv(path+"imputed.allWhites.50-0.0.chr22.csv.gz", sep = " ", compression = 'gzip')
hla = pd.read_csv(path+"hla.50-0.0.csv.gz", sep = " ", compression = 'gzip')
hla_omnibus = pd.read_csv(path+"hla.omnibus.50-0.0.csv.gz", sep = " ", compression = 'gzip')

genotyped_nr_res = pd.read_csv(path+"genotyped.normRank.allWhites.50-0.0.chr22.csv.gz", sep = " ", compression = 'gzip')
imputed_nr_res = pd.read_csv(path+"imputed.normRank.allWhites.50-0.0.chr22.csv.gz", sep = " ", compression = 'gzip')


gen_stats = pd.read_csv(path+"snps.genotyped.chr22.csv", sep = " ")
imp_stats = pd.read_csv(path+"snps.imputed.chr22.csv", sep = " ")
hla_freq = pd.read_csv(path+"hla.freq", sep = " ")


