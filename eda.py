#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:47:03 2019

@author: raphaelsinclair
"""


# ============= LOAD DATA =============== #

import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


## Load dataframes
#hpc_path = '/rdsgpfs/general/project/medbio-berlanga-group/live/projects/ml_trait_prediction'
#os.chdir(hpc_path)
#path = os.path.join(hpc_path, directory)
os.chdir(os.path.expanduser('~'))
home = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/'
directory = 'Data/Processed/'

data = pd.read_pickle(
        os.path.join(home, directory) + 'processed_data_sample.pkl')
int_df = pd.read_pickle(
        os.path.join(home, directory) + 'integrated_data_sample.pkl')



# ============= EXPLORATORY ANALYSIS =============== #


##### FUNNEL PLOTS (BETA, SE)

gs1 = gridspec.GridSpec(2, 2, height_ratios=[10, 8])
gs1.update(wspace = 0.3, hspace = 0.3)
fig1 = plt.figure()

sns.set_style('darkgrid')


# Funnel plot
ax1 = fig1.add_subplot(gs1[0, :])
sns.scatterplot(int_df['Beta'], int_df['SE'], ax = ax1)

beta_xlim = max(np.abs(int_df['Beta']))
se_ylim = np.ceil(max(int_df['SE']))
mean_beta = np.mean(int_df['Beta'])

ax1.set(xlim = (-beta_xlim, beta_xlim), ylim = (0, se_ylim))
ax1.invert_yaxis()
ax1.axvline(mean_beta, linestyle = '--', c = 'grey')
ax1.text(mean_beta, 7,
         r'$\mu={:.{}f}$'.format(mean_beta, 4),
         fontsize = 10)


# Violin plot
ax2 = fig1.add_subplot(gs1[1, 0])
sns.violinplot(y=int_df['SE'], ax = ax2)


# Distribution plot
ax3 = fig1.add_subplot(gs1[1, 1])
sns.distplot(int_df['Beta'], ax = ax3)

ax3.set(xlim = (-beta_xlim, beta_xlim))
ax3.set_ylabel('Distribution', fontsize = 10)


plt.show()
# Funnel shows little/no publication bias (within sample)



##### DISTRIBUTION PLOTS (ISCORES, HWE-P)

gs2 = gridspec.GridSpec(2, 1)
gs2.update(hspace = 0.5)
fig2 = plt.figure()

sns.set_style('darkgrid')


# Iscores
ax4 = fig2.add_subplot(gs2[0, 0])
sns.distplot(int_df['iscores'], hist = False, ax = ax4)

percentile = np.percentile(int_df['iscores'], 5)
iscore_prop = sum(int_df['iscores'] >= percentile)/len(int_df['iscores'])
iscore_xkde, iscore_ykde = ax4.lines[0].get_data()
ax4.text(0.7, 2, '{:.1f}%'.format(100 * iscore_prop))

ax4.set(xlim = (0, 1))
#ax4.set_xticks(sorted(ax4.get_xticks().tolist() + [percentile]))
ax4.axvline(percentile, linestyle = '--', c = 'red')
ax4.text(percentile, 0, '{:.2f}'.format(percentile))

ax4.fill_between(iscore_xkde, iscore_ykde, where=(iscore_xkde >= percentile),
                 interpolate = True, alpha = 0.5)
ax4.fill_between(iscore_xkde, iscore_ykde, where=(iscore_xkde <= percentile),
                 interpolate = True, alpha = 0.25, color = 'red')


# HWE_P
ax5 = fig2.add_subplot(gs2[1, 0])
sns.distplot(int_df['HWE_P'], hist = False, ax = ax5)

hwe_xkde, hwe_ykde = ax5.lines[0].get_data()
threshold = 0.05
hwe_prop = sum(int_df['HWE_P'] > threshold)/len(int_df['HWE_P'])

ax5.set(xlim = (0, 1))
x = ax5.get_xlim
ax5.axvline(threshold, linestyle = '--', c = 'orange')
ax5.text(threshold, 2, 'cut-off = {}'.format(threshold))
ax5.text(0.7, 1, '{:.1f}%'.format((100 * hwe_prop)))

ax5.fill_between(hwe_xkde, hwe_ykde, where=(hwe_xkde >= threshold),
                 interpolate = True, alpha = 0.5)
ax5.fill_between(hwe_xkde, hwe_ykde, where=(hwe_xkde <= threshold),
                 interpolate = True, alpha = 0.25, color = 'orange')


plt.show()



##### P_VALUE DISTRIBUTION PLOTS

gs3 = gridspec.GridSpec(1, 2)
gs3.update(wspace = 0.5)
fig3 = plt.figure()

sns.set_style('darkgrid')


# P_value
ax6 = fig3.add_subplot(gs3[0, 0])
sns.distplot(int_df['p_value'], hist = False, 
             kde_kws={'shade' : True}, ax = ax6)

ax6.set(xlim = (0, 1))


# Log-transformed p_value
ax7 = fig3.add_subplot(gs3[0, 1])
sns.distplot(int_df['log_p_val'], hist = False, 
             kde_kws={'shade' : True}, ax = ax7)

ax7.set(xlim = (0, 1))


plt.show()
# NOTE: p_values should be uniformly distributed. Log_transformation typically
# shows a peak towards zero in GWAS



##### DISTRIBUTION PLOTS (POSITION, MAF)
# Position
fig4 = sns.distplot(int_df['Position'], hist = False, 
             kde_kws={'shade' : True})
fig4.set(xlim = (0, max(int_df['Position'])))


plt.show()
# Should be uniform. 

# Mean allele frequency
fig5 = sns.distplot(int_df['MAF'], hist = False, 
             kde_kws={'shade' : True})
fig5.set(xlim = (0, max(int_df['MAF'])))


plt.show()
# Majority of SNPs have allele frequency close to 0



##### CATEGORICAL BARPLOTS

# Define numerical and categorical variables
#numeric_vars = int_df.select_dtypes(
#        exclude=['object', 'category']).columns.to_list()
categorical_vars = int_df.select_dtypes(
        include=['category']).columns.to_list()


sns.set_style('darkgrid')


ncols = 3
if len(categorical_vars) % ncols == 0:
    nrows = len(categorical_vars)//ncols
else: 
    nrows = len(categorical_vars)//ncols + 1
    
gs4 = gridspec.GridSpec(nrows, ncols)
gs4.update(wspace = 0.5, hspace = 0.5)

fig6 = plt.figure()
for index, column in enumerate(categorical_vars):
    axes = fig6.add_subplot(gs4[index // ncols, index % ncols])
    sns.countplot(int_df[column].dropna(),
                 ax = axes)    
    axes.set_xlabel(column)
    axes.set_ylabel('Counts')


plt.show()

all(int_df['Allele_v2'] == int_df['A2_v2'])
## Confirms Allele_v2 = A2_v2. Consider removing Allelle_v2 from preprocessing!

###NOTE: ADD %AGES ABOVE BARS



##### STRATIFIED DISTPLOTS

# P_value distribution by minor/major allele
gs5 = sns.FacetGrid(int_df, col = 'A1_v2', row = 'A2_v2', height = 2,
                    margin_titles = True)
fig7 = gs5.map(sns.distplot, 'p_value', hist = False,
               kde_kws={'shade' : True})

fig7.set(xlim = (0,1))
#fig7.set_title('P_value stratified by Allele')

## CHANGE DIAGONAL PLOTS
#gs5.map_diag(sns.scatterplot, 'Beta', 'SE')



plt.show()


# Log_transformed p_values
gs6 = sns.FacetGrid(int_df, col = 'A1_v2', row = 'A2_v2', height = 2,
                    margin_titles = True)
fig8 = gs6.map(sns.distplot, 'log_p_val', hist = False,
               kde_kws={'shade' : True})

fig8.set(xlim = (0, None))


plt.show()



##### CORRELATION PLOT

# Generate correlation matrix and mask for the upper triangle
corr = int_df.drop('log_p_val', axis = 1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask, k = 0)] = True
cmap = sns.diverging_palette(240, 10, as_cmap=True)

fig9 = sns.heatmap(corr, mask = mask, center = 0,
            vmax = 0.5, cmap = cmap)


plt.show()






##### 2-WAY CROSS TABLE

# Create 2-way heatmap for A1 vs. A2 categories
rows = int_df['A1_v2'].cat.categories
cols = int_df['A2_v2'].cat.categories
al_matrix = pd.crosstab(int_df['A1_v2'], int_df['A2_v2'], margins = False).values
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask, k = 0)] = True

#cmap = sns.diverging_palette(240, 10, as_cmap=True)
#fig10 = sns.heatmap(al_matrix, mask = mask, center = 0, cmap = cmap)
#
#
#plt.show()





###### MANHATTTAN PLOT
#
## Order values and set up manhattan plot
#manh_vals = int_df.sort_values(
#        by = ['Chr_no', 'Position'],
#        ascending=[True, True])[['Chr_no', 'Position','log_p_val']]
#manh_vals['ind'] = range(len(manh_vals))
#
#
#fig10 = plt.figure()
#ax8 = fig10.add_subplot(111)
##colors = ['red','green','blue', 'yellow']
#
#x_labels = []
#x_labels_pos = []
#
#for num, (name, group) in enumerate(manh_vals[0:5]):
#    print(num)
#    print((name, group))
#    
#    
#    
#    group.plot(kind='scatter', x='ind', y='minuslog10pvalue',
#               color=colors[num % len(colors)], ax=ax)
#    x_labels.append(name)
#    x_labels_pos.append(
#            (group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
#
#
##ax.set_xticks(x_labels_pos)
##ax.set_xticklabels(x_labels)
##ax.set_xlim([0, len(df)])
##ax.set_ylim([0, 3.5])
##ax.set_xlabel('Chromosome')

