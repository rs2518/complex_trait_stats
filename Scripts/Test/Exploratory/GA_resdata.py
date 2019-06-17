#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:03:52 2019

@author: raphaelsinclair
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns


# Load data and rename columns
# Tab separated instead of comma separated
path = "~/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/clinical_c_Block_J40-J47/"
df_imputed = pd.read_csv(path+"imputed.allWhites.clinical_c_Block_J40-J47.chr22.csv", sep = " ")

df_genotyped = pd.read_csv(path+'genotyped.allWhites.clinical_c_Block_J40-J47.chr22.csv', sep = " ")
df_genotyped.columns = ['SNP', 'Allele', 'Beta', 'SE', 'P_value']


merged_df = pd.merge(df_imputed, df_genotyped, right_on = 'SNP',
                     left_on = 'SNP', how='outer')



##### COLUMNS #####
# SNP = SNP id
# Position = Position along chromosome given as count
# A1 = Allele 1
# A2 = Allele 2
# MAF = Minor Allele Frequency (See GeneAtlas FAQs)
# HWE-P = Probability that a genotype falls under Hardy-Weinberg Equilibrium and thus monogenic? (See GeneAtlas FAQs)
# iscore = ?


# ==== Check data ==== #

# Check if any SNPs in dataframe are duplicated. Set as rownames
if len(merged_df.duplicated(subset=['SNP'], keep=False)[merged_df.duplicated(subset=['SNP'],
             keep=False) == True]) == 0:
    print('No duplicated SNPs')
    

# Check for incorrect/unexpected entries
print(merged_df['A1'].value_counts(dropna = False))
print(merged_df['A2'].value_counts(dropna = False))


#sns.distplot(merged_df['Position'])
#plt.show()
## Uniform. Distances appear to be unique

#sns.distplot(df_imputed['MAF'])
#plt.show()
## Majority of SNPs have allele frequency of 0 (close to 0)


# Summary statistics and plots
print(merged_df.describe())


merged_df['A1'] = df_imputed['A1'].astype('category')
A1_categories = list(merged_df['A1'].cat.categories) + ['Missing']
A1_counts = list(merged_df['A1'].value_counts(dropna = False))
plt.bar(A1_categories, A1_counts, color = ['red', 'yellow', 'blue', 'green', 'grey'],
        edgecolor = 'black')
plt.xlabel('A1')
plt.ylabel('Counts')
plt.show()


merged_df['A2'] = merged_df['A2'].astype('category')
A2_categories = list(merged_df['A2'].cat.categories) + ['Missing']
A2_counts = list(merged_df['A2'].value_counts(dropna = False))
plt.bar(A2_categories, A2_counts, color = ['magenta', 'orange', 'cyan', 'lime', 'grey'],
        edgecolor = 'black')
plt.xlabel('A2')
plt.ylabel('Counts')
plt.show()



# ==== Data preparation ==== #

# Convert categorical variables into binary variables
binary_al = pd.get_dummies(merged_df[['A1', 'A2']], columns = ['A1', 'A2'], prefix = ['A1_','A2_'])


# Scale continuous data and create dataframe of predictors
from sklearn.preprocessing import StandardScaler

continuous_vars = merged_df.drop(['A1', 'A2'], axis = 1)
standard_scaler = StandardScaler()
scaled_cont_data = pd.DataFrame(standard_scaler.fit_transform(continuous_vars),
                           columns = continuous_vars.columns,
                           index = continuous_vars.index)

Class = pd.Series(np.where(df_imputed['iscore'] > 0.85, 1, 0),
                  index = continuous_vars.index, name = 'Class')

processed_data = pd.concat([scaled_cont_data, binary_al, Class], axis = 1)


# Assign labels
X = processed_data.drop(['Class'], axis = 1).values
y = processed_data['Class'].values


# Train_test_split function
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 42,
                                                    stratify = y)


print(np.array(np.unique(y_test, return_counts=True)))
print(12 * '-')
print(np.array(np.unique(y_train, return_counts=True)))


# ==== Testing models ==== #

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, roc_curve


folds = 10


##### Logistic Regression

#logreg = LogisticRegression(solver = 'sag')
#parameters = {'penalty': ['none']}
#base_lr_cv = GridSearchCV(logreg, parameters, cv = folds, scoring = 'roc_auc')
#base_lr_cv.fit(X_train, y_train)
#
#print("Tuned Logistic Regression Parameters: {}".format(base_lr_cv.best_params_)) 
#print("Best training AUC score is {}".format(base_lr_cv.best_score_))
#y_pred_base_lr = base_lr_cv.predict(X_test)
#print(12*'-')
#print("AUC score on test set: {}".format(base_lr_cv.score(X_test, y_test)))
#print(12*'-')
#print(classification_report(y_test, y_pred_base_lr))

base_lr = SGDClassifier(loss = 'log', random_state = 21)
parameters = {'penalty': ['none']}
base_lr_cv = GridSearchCV(base_lr, parameters, cv = folds, scoring = 'roc_auc')
base_lr_cv.fit(X_train, y_train)
 
print("Best training AUC score is {}".format(base_lr_cv.best_score_))
y_pred_base_lr = base_lr_cv.predict(X_test)
print(12*'-')
print("AUC score on test set: {}".format(base_lr_cv.score(X_test, y_test)))
print(12*'-')
print(classification_report(y_test, y_pred_base_lr))

# ROC curve
y_pred_base_prob = base_lr_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_base_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



##### Perceptron

perceptron = Perceptron(random_state = 21)
parameters = {'penalty': ['none']}
perceptron_cv = GridSearchCV(perceptron, parameters, cv = folds)
perceptron_cv.fit(X_train, y_train)

print("Best training score is {}".format(perceptron_cv.best_score_))
y_pred_per = perceptron_cv.predict(X_test)
print(12*'-')
print("Score on test set: {}".format(perceptron_cv.score(X_test, y_test)))
print(12*'-')
print(classification_report(y_test, y_pred_per))

#parameters = {'logistic__C': c_space, 'logistic__penalty': ['l1', 'l2']}






## Determine association/significance from p-values
#alpha = 0.05
#bonf = alpha/n
#df_imputed['Association?'] = np.where(df_imputed['HWE-P'] > bonf, 0,1)














