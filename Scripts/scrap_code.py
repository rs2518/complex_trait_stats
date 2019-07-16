#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:45:42 2019

@author: raphaelsinclair
"""


#import seaborn as sns
#import matplotlib.pyplot as plt


##SAVE IN LOCATION RELATIVE TO SCRIPT
import sys
#script_name = sys.argv[0]     # import script name from bash
script_name = "GA_preprocessing_v2.py"
for root, dirs, files in os.walk(os.path.expanduser('~')):
    for file in files:
        if file.endswith(script_name):
            filename = file
            path = root
            break
if not filename.endswith(script_name):
    print('Could not find specified file')
else:
    os.chdir(path)
    print('Directory changed to :', path)
    
    if not os.path.exists('./Results'):
        os.mkdir('./Results')
        print("Created './Results/' directory")
    else:    
        print("'./Results/' directory already exists" )
        preprocessed_data.to_csv(os.path.join('./Results', 
                                           'processed_data.csv'))
        merged_data.to_csv(os.path.join('./Results', 
                                           'integrated_data.csv'))
        



# EXTRACT NUMBER OF ROWS WITHOUT SAVING DATA
count = 0
for line in open(thefilepath).xreadlines(  ): count += 1




## SET LAST COLUMN TO BE FIRST
add_cols = ['p_value', 'SNP', 'Position']
processed_data = pd.concat([
        scaled_data, dummy_data,
        merged_data[add_cols]], axis = 1)
processed_data = processed_data[
        processed_data.columns[-(len(add_cols) - 1):].to_list() + processed_data.columns[:-(len(add_cols) - 1)].to_list()]


#continuous_vars = ['iscores', 'Beta', 'SE', 'MAF', 'HWE_P']
#categorical_vars = [col + '_v2' for col in al_vars] + ['Chr_no']

#merged_data[col + '_v2'] = pd.Categorical(merged_data[col], 
#        categories = al_categories, ordered = False)




## SAVE DATA AS CSV
if not os.path.exists('Processed'):
    os.mkdir('Processed')
    print('Created \'./Processed/\' directory')
    sample_processed.to_csv(os.path.join('./Processed', 
                                       'processed_data_sample.csv'))
    sample_merged.to_csv(os.path.join('./Processed', 
                                       'integrated_data_sample.csv'))
else:    
    print('\'./Processed/\' directory already exists')
    sample_processed.to_csv(os.path.join('./Processed', 
                                       'processed_data_sample.csv'))
    sample_merged.to_csv(os.path.join('./Processed', 
                                       'integrated_data_sample.csv'))




## LOAD CSV DATA
data = pd.read_csv(
        os.path.join(home, directory) + 'processed_data_sample.csv',
        index_col = 0)
int_df = pd.read_csv(
        os.path.join(home, directory) + 'integrated_data_sample.csv',
        index_col = 0)







## CORRELATION PLOT
C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(C_mat, vmax = .8, square = True)
# Could set vmax to be median correlation score
# i.e. Take list of unique correlation scores and calculate median
# (or other suitable percentile depending on distribution)
plt.show()




## COMPLEX SUBPLOTS ON SAME GRAPH
# SEE https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()



## VISUALISE DISTRIBUTIONS OF ALL NUMERIC VARIABLES USING SUBPLOTS
# (UNUSED SUBPLOTS REMOVED)
ncols = 3
if len(numeric_vars) % ncols == 0:
    nrows = len(numeric_vars)//ncols
else: 
    nrows = len(numeric_vars)//ncols + 1
    
gs = gridspec.GridSpec(nrows = nrows, ncols = ncols)
fig = plt.figure()
for index, column in enumerate(numeric_vars):
    axes = fig.add_subplot(gs[index // ncols, index % ncols])
    sns.distplot(int_df[column].dropna(),
                 ax = axes)    
    axes.set_xlabel(column)
    axes.set_ylabel('Density : {}'.format(column))
plt.show()
##### NOTE: NOT PARTICULARLY HELPFUL TO VIEW THIS WAY. VIEW AND
##### ANALYSE VARIABLES INDIVIDUALLY




## STRATIFIED SCATTERPLOT
# Create scatter plots
g = sns.FacetGrid(tips, col="sex", row="smoker", margin_titles=True)
g.map(sns.plt.scatter, "total_bill", "tip")

# Add a title to the figure
g.fig.suptitle("this is a title")

# Show the plot
plt.show()




## FUNNEL PLOT
sns.set_style('ticks')
ax = sns.scatterplot(int_df['Beta'], int_df['SE'])
ax.set(xlim = (-max(np.abs(int_df['Beta'])), max(np.abs(int_df['Beta']))))
#ax.set(xlim = (-max(np.abs(int_df['Beta'])), max(np.abs(int_df['Beta']))),
#       ylim = (0, max(int_df['SE']) + 1))     # Set y-axis as well
ax.axvline(np.mean(int_df['Beta']), linestyle = '--', c = 'grey')
ax.set_title('Funnel Plot', fontsize = 12, fontweight = 'bold')
#plt.ylim(reversed(plt.ylim()))
ax.invert_yaxis()
ax.text(np.mean(int_df['Beta']), 7,
         r'$\mu={:.{}f}$'.format(np.mean(int_df['Beta']), 4),
         fontsize = 10)
plt.show()




## 2-WAY CROSS TABLE
rows = int_df['A1_v2'].cat.categories
cols = int_df['A2_v2'].cat.categories
al_matrix = pd.crosstab(int_df['A1_v2'], int_df['A2_v2'], margins = False).values

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask, k = 0)] = True

#cmap = sns.diverging_palette(240, 10, as_cmap=True)
fig10 = sns.heatmap(al_matrix, mask = mask, center = 0, cmap = cmap)


plt.show()



## SUBPLOT FOR CROSS-VALIDATION
## Training scores plots
#
#figtest, axes = plt.subplots(1,2, sharey = True)
#
#sns.set_style('darkgrid')
#
#for split in range(folds):
#    train_scores = -results['split%s_train_score' % (split)]
#    axes[0].semilogx(l1_space,
#                 train_scores,
#                 linestyle = '--',
#                 label = 'Fold {0}'.format(split + 1))
#
#axes[0].set_xlim([l1_space[0], l1_space[-1]])
#axes[0].set_ylabel('Mean Squared Error')
#axes[0].set_xlabel('alpha (L1 penalty)')
#axes[0].set_title('Training scores by fold', fontweight = 'bold')
#axes[0].legend()
# 
#  
##plt.show()
#
#
## Plot mean train scores with confidence intervals
#mean_train_scores = -results['mean_train_score']
#std_train_scores = results['std_train_score']
#
#axes[1].semilogx(l1_space, mean_train_scores)
#axes[1].semilogx(l1_space, mean_train_scores + std_train_scores,
#             linestyle = '--', color = 'navy')
#axes[1].semilogx(l1_space, mean_train_scores - std_train_scores,
#             linestyle = '--', color = 'navy')
#
#axes[1].set_ylabel('Mean Squared Error')
#axes[1].set_xlabel('alpha (L1 penalty)')
#axes[1].axhline(np.max(mean_train_scores), linestyle = ':', color='.5')
#axes[1].text(l1_space[1],
#          np.max(mean_train_scores) - 0.5 * np.max(std_train_scores),
#          'Highest training score = {:.3g}'.format(np.max(mean_train_scores)),
#          color = 'g')
#axes[1].set_xlim([l1_space[0], l1_space[-1]])
#
#axes[1].fill_between(l1_space, mean_train_scores + std_train_scores,
#                 mean_train_scores - std_train_scores, alpha=0.2)
#axes[1].set_title('Mean training scores +/- standard error', fontweight = 'bold')
#
#
#plt.show()


