#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:34:07 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



# ============= DATA PREPARATION =============== #

# Load dataframes
path = '/Users/raphaelsinclair/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/'
data = pd.read_csv(path + 'processed_data.csv')
df = pd.read_csv(path + 'integrated_data.csv', index_col = 0)


# Set input/target values
X = data.drop(['SNP'], axis = 1)
y = df['p_value']


# View distribution of target (view impact of transformation)
sns.distplot(y)
plt.show()

sns.distplot(-np.log(y))
plt.show()


# Split data into test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 21)


# Log transform p_values
log_y_train = -np.log(y_train)
log_y_test = -np.log(y_test)


# ============= LINEAR MODEL =============== #

# Scikit-learn linear model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Fit model and predict on test set. Return metrics
modela = linear_model.LinearRegression()
modela.fit(X_train, y_train)
y_pred = modela.predict(X_test)

modela_coefs = pd.DataFrame(np.append(modela.intercept_, modela.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients :\n',modela_coefs)
print('Mean-squared error :', mean_squared_error(y_test, y_pred))
print('Variance explained (R^2) :' , r2_score(y_test, y_pred))


plt.scatter(x = X_test.index, y = (y_test - y_pred), c = 'g')
plt.show()     # Plot not useful

sns.distplot((y_test - y_pred))
plt.show()     # Residuals should be normally distributed


# Log transformed p-values
modelb = linear_model.LinearRegression()
modelb.fit(X_train, log_y_train)
log_y_pred = modelb.predict(X_test)

modelb_coefs = pd.DataFrame(np.append(modelb.intercept_, modelb.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients (log-transformed) :\n',modelb_coefs)
print('Mean-squared error :', mean_squared_error(log_y_test, log_y_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_y_pred))


plt.scatter(x = X_test.index, y = (log_y_test - log_y_pred), c = 'g')
plt.show()     # Plot not useful

sns.distplot((log_y_test - log_y_pred))
plt.show()     # Residuals should be normally distributed


##########


## Statsmodels linear model
#from statsmodels.formula.api import glm, ols
#import statsmodels.api as sm
#
#
## Concatenate data with targets
#train_data = pd.concat([X_train, y_train], axis = 1)
#test_data = pd.concat([X_test, y_test], axis = 1)
#log_train_data = pd.concat([X_train, log_y_train], axis = 1)
#log_test_data = pd.concat([X_test, log_y_test], axis = 1)
#
#
#formula = y.name + '~' + X.columns[0]
#for index in range(1, len(X.columns)):
#    formula = formula + '+' + str(X.columns[index])
#
#
#test = 'p_value~iscores+Beta+SE+MAF+Allele_v2__A+Allele_v2__C+Allele_v2__G+Allele_v2__T+A1_v2__A+A1_v2__C+A1_v2__G+A1_v2__T+A2_v2__A+A2_v2__C+A2_v2__G+A2_v2__T+Chr_no__1+Chr_no__2+Chr_no__3+Chr_no__4+Chr_no__5+Chr_no__6+Chr_no__7+Chr_no__8+Chr_no__9+Chr_no__10+Chr_no__11+Chr_no__12+Chr_no__13+Chr_no__14+Chr_no__15+Chr_no__16+Chr_no__17+Chr_no__18+Chr_no__19+Chr_no__20+Chr_no__21+Chr_no__22'
## NO 'HWE-P' because '-' sign
#
#glm = glm(formula = test, 
#          data = train_data, 
#          family = sm.families.Gaussian()).fit()
#
#print('Linear model coefficients :\n',glm.params)
#
#
## Log transformed p-values
#glmb = glm(formula = test, 
#          data = log_train_data, 
#          family = sm.families.Gaussian()).fit()
#
#print('Linear model coefficients :\n',glmb.params)


# ============= NEURAL NETWORK =============== #

# Keras neural network
from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model
from keras.utils import plot_model


# Create layers
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)


# Build model and plot visual image of model
nn_model = Model(input_tensor, output_tensor)
nn_model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
print(nn_model.summary())

#plot_model(nn_model, to_file=path+'nn_model.png')
#image = plt.imread(path+'nn_model.png')
#plt.imshow(data)
#plt.show()


## Fit and evaluate model
#nn_model.fit(------, y, 
#              epochs=1, batch_size=128, validation_split=0.1, verbose=True)




