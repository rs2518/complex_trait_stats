## Statistical learning of genomic variation associated with complex traits
MSc Health Data Analytics w/ Machine Learning Term 3 research project

Dr. Antonio Berlanga, Dr. Deborah Scheider-Luftman, Raphael Sinclair

### The 'complex_trait_stats' repository
The project directory is structured as follows:
```
README.md
__init__.py
data/
   |-- README.txt
   |-- annovar/
   |   |-- 0_setup_annovar.sh
   |   |-- 1_annotate_data.sh
   |   |-- README.txt
   |-- geneatlas/
   |   |-- 0_vi_downloads.sh
   |   |-- 1_trait_download.sh
   |   |-- README.txt
   |   |-- Traits_Table_GeneATLAS.csv
eda/
   |-- eda.py
models/
   |-- __init__.py
   |-- _linear_regression.py
   |-- _neural_network.py
   |-- _partial_least_sq.py
   |-- _penalised_regression.py
   |-- _random_forest.py
   |-- experimental/
   |   |-- __init__.py
   |   |-- exp_lin_reg.py
   |   |-- exp_nn.py
   |   |-- exp_pen_reg.py
   |   |-- exp_pls.py
   |   |-- exp_rf.py
   |-- models.py
preprocess/
   |-- process_anno.py
   |-- process_geneatlas.py
setup.yml
utils.py
```
