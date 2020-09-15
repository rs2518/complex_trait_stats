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

# ANNOVAR leaves lots of missing data. Consider another source?