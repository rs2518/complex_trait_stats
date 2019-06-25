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




# PLOT DISTRIBUTION OF P_VALUES
# View distribution of labels (i.e. 'p_value')
sns.distplot(merged_data['p_value'].dropna())
plt.show()

# View distribution of target (view impact of transformation)
sns.distplot(y)
plt.show()

sns.distplot(-np.log(y))
plt.show()


formula = y_train.name + '~' + processed_cols[0]
for index in range(1, len(processed_cols)):
    if data[processed_cols[index]].dtype != np.float:
        formula = formula + '+c(' + str(processed_cols[index] + ')')
    else:
        formula = formula + '+' + str(processed_cols[index])





print([i for i in processed_cols if data[i].dtype != np.float]



