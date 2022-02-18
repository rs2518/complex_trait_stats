## Statistical learning of genomic variation associated with complex traits
MSc Health Data Analytics w/ Machine Learning Term 3 research project

Dr. Antonio Berlanga, Dr. Deborah Scheider-Luftman, Raphael Sinclair


## Abstract
#### Introduction
Complex traits make up a vast majority of human phenotypes - particularly those responsible for disease outcomes. These types of traits produce the continuous ranges of phenotypic values seen among populations of organisms, but the underlying mechanisms responsible for this spectrum of variability are still a mystery.

#### Background
Tools such as Mendelian Randomisation have helped researchers successfully identify causal genetic pathways for complex phenotypes, meanwhile machine learning frameworks have been able to predict patient disease risk as well as the functional effects of some genetic mechanisms. Studies have also found that complex traits are mostly driven by non-coding genetic variation, however, understanding complex traits on a genetic level still remains a challenging task. In this pilot study, we will use machine learning to predict genomic variation that is highly associated to complex traits and characterise these findings with biological knowledge.

#### Methods
We extracted summary-level GWAS data from 22,000 randomly selected SNPs for the standing height trait and trained a series of machine learning models to recover the log-transformed p-values for phenotypic association. Multiple feature importance techniques were implemented to highlight the predictors that heavily contribute to the model outcomes. We validated our models and assessed the reliability of their outcomes using two newly developed methodologies - the positive control and negative control validation strategies.

#### Results
Our study showed that the random forest was the most effective method for predicting complex trait associated genetic variation (Q<sup>2</sup> = 0.505), meanwhile the multilayer perceptron was least effective (Q<sup>2</sup> = 0.032). The linear-based models consistently detected a linear association between minor allele frequency and a genetic variant’s association to the standing height trait. Our negative control validation procedure revealed that all the models could detect true positive associations with 100% accuracy whilst rejecting over 93.4% of false positives - although the random forest was slightly more prone to false positives with a rejection rate of 92%. The positive control validation procedure showed that all models were able to identify the positive control feature (an artificial feature that is associated to the target by construction) at least 96.5% of the time in the presence of Gaussian noise up until a standard deviation of 25 (σ = 25). Beyond this point, the detection rate deteriorates rapidly for all except the random forest model.

#### Discussion
This pilot study provides a heuristic machine learning framework for predicting complex trait associated genetic variation based on summary-level GWAS data. Though the random forest was best suited to this regression task, further research will be necessary to fully characterise the patterns of association among the complex trait associated variants. Therefore, we propose the addition of other available biomedical data (e.g. annotation databases) as well as a deeper exploration of machine learning methods on this data for future works.


## The 'complex_trait_stats' repository
The project directory is structured as follows:
```
.gitignore
README.md
cts
   |-- README.txt
   |-- __init__.py
   |-- models
   |   |-- README.txt
   |   |-- __init__.py
   |   |-- _linear_regression.py
   |   |-- _neural_network.py
   |   |-- _partial_least_sq.py
   |   |-- _penalised_regression.py
   |   |-- _random_forest.py
   |   |-- experimental
   |   |   |-- __init__.py
   |   |   |-- exp_lin_reg.py
   |   |   |-- exp_nn.py
   |   |   |-- exp_pen_reg.py
   |   |   |-- exp_pls.py
   |   |   |-- exp_rf.py
   |-- utils.py
data
   |-- README.txt
   |-- annovar
   |   |-- 0_setup_annovar.sh
   |   |-- 1_annotate_data.sh
   |   |-- README.txt
   |-- geneatlas
   |   |-- 0_vi_downloads.sh
   |   |-- 1_trait_download.sh
   |   |-- README.txt
   |   |-- Traits_Table_GeneATLAS.csv
eda.py
feature_selection.py
model_diagnostics.py
negative_control_validation.py
perm_rf.py
permutation_importances.py
plot_permutation_results.py
plot_validation_results.py
pos_ctrl_mlp.py
positive_control_validation.py
preprocess
   |-- README.txt
   |-- _process_pipe.py
   |-- process_anno.py
   |-- process_geneatlas.py
   |-- process_giant.py
rf_importances.py
setup.yml
tune_lr.py
tune_mlp.py
tune_pls.py
tune_pr.py
tune_rf.py
```
