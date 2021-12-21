"""
Created on Sat Dec  4 15:25:27 2021

@author: murta

Heavily optimized and cleaned by kadenk
"""

import os
import numpy as np
from util.average_comparisons.dataset import *
from util.dataset import *
from util.statistics import confusion_matrix, plot
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso


cache_dir = os.path.join("cache", os.path.basename(__file__).split(".")[0])
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)

################################################################################
#model features

csv_filename = "dataset.csv"
csv_filename = os.path.join(cache_dir, csv_filename)
if not os.path.exists(csv_filename):
	constructDataset(csv_filename)

df = readTable(csv_filename)

df = balanceDataset(df)
x = df[df.columns.drop(['season', 'points0', 'points1'])]
x = standardizeDataset(x)
print(x)
y = (df['points0'] - df['points1'])

# Split the dataset 80/20.
x_train_validation, x_test, y_train_validation, y_test = train_test_split(x, y, 0.8)
x_train_validation = x_train_validation.to_numpy()
x_test  = x_test.to_numpy()
y_train_validation = y_train_validation.values.astype(float)
y_test  = y_test.values.astype(float)


################################################################################
# OLS CV
model = LinearRegression()

k = 4
kf = KFold(n_splits=k, random_state=None)

mse_min = None
best_model = None

for train_index, validation_index in kf.split(x_train_validation):
	x_train, x_validation = x_train_validation[train_index,:], x_train_validation[validation_index,:]
	y_train, y_validation = y_train_validation[train_index], y_train_validation[validation_index]

	model.fit(x_train, y_train)
	validation_predictions = model.predict(x_validation).reshape((-1,1))

	mse = mean_squared_error(y_validation, validation_predictions)
	if mse_min is None or mse < mse_min:
		mse_min = mse
		best_model = model

test_predictions = best_model.predict(x_test).reshape((-1,1))

confusion_matrix(test_predictions, y_test, lambda y: y > 0)

title = "OLS"
plot(y_test, test_predictions, title, os.path.join(cache_dir, "plot_{0}.png".format(title).replace(" ", "")))

################################################################################
# Ridge

fit_intercept = True 

# Take the train/test data and perform k-fold CV with it and record the model and mse
k = 4 
kf = KFold(n_splits=k, random_state=None)

mse_min = None
best_alpha = None

for train_index, validation_index in kf.split(x_train_validation):
	x_train, x_validation = x_train_validation[train_index,:], x_train_validation[validation_index,:]
	y_train, y_validation = y_train_validation[train_index], y_train_validation[validation_index]

	alpha_list = np.linspace(0.0000001, 15, 1000)

	for alpha in alpha_list:
		model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=10e6)
		model.fit(x_train, y_train)
		validation_predictions = model.predict(x_validation).reshape((-1,1))

		mse = mean_squared_error(y_validation, validation_predictions)
		if mse_min is None or mse < mse_min:
			mse_min = mse
			best_alpha = alpha

#######################################################################
# Train the model using all the 80% data
model = Ridge(alpha=best_alpha, fit_intercept=fit_intercept, max_iter=10e6)
model.fit(x_train_validation, y_train_validation)
test_predictions = model.predict(x_test).reshape((-1,1))


confusion_matrix(test_predictions, y_test, lambda y: y > 0)

title = "Ridge Regression"
plot(y_test, test_predictions, title, os.path.join(cache_dir, "plot_{0}.png".format(title).replace(" ", "")))


################################################################################
# Lasso

fit_intercept = True 

# Take the train/test data and perform k-fold CV with it and record the model and mse
k = 4 
kf = KFold(n_splits=k, random_state=None)

mse_min = None
best_alpha = None

for train_index, validation_index in kf.split(x_train_validation):
	x_train, x_validation = x_train_validation[train_index,:], x_train_validation[validation_index,:]
	y_train, y_validation = y_train_validation[train_index], y_train_validation[validation_index]

	alpha_list = np.linspace(0.0000001, 15, 1000)

	for alpha in alpha_list:
		model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10e6)
		model.fit(x_train, y_train)
		validation_predictions = model.predict(x_validation).reshape((-1,1))

		mse = mean_squared_error(y_validation, validation_predictions)
		if mse_min is None or mse < mse_min:
			mse_min = mse
			best_alpha = alpha

#######################################################################
# Train the model using all the 80% data
model = Lasso(alpha=best_alpha, fit_intercept=fit_intercept, max_iter=10e6)
model.fit(x_train_validation, y_train_validation)
test_predictions = model.predict(x_test).reshape((-1,1))


confusion_matrix(test_predictions, y_test, lambda y: y > 0)

title = "Lasso Regression"
plot(y_test, test_predictions, title, os.path.join(cache_dir, "plot_{0}.png".format(title).replace(" ", "")))


################################################################################
# Elastic Net

fit_intercept = True 

# Take the train/test data and perform k-fold CV with it and record the model and mse
k = 4 
kf = KFold(n_splits=k, random_state=None)

mse_min = None
best_alpha = None
best_l1_ratio = None

for train_index, validation_index in kf.split(x_train_validation):
	x_train, x_validation = x_train_validation[train_index,:], x_train_validation[validation_index,:]
	y_train, y_validation = y_train_validation[train_index], y_train_validation[validation_index]

	l1_ratio_list = np.linspace(0.0001, 0.9999, 5)
	alpha_list = np.linspace(0.0000001, 5, 1000)

	for alpha in alpha_list:
		for l1_ratio in l1_ratio_list:
			model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=10e6)
			model.fit(x_train, y_train)
			validation_predictions = model.predict(x_validation).reshape((-1,1))

			mse = mean_squared_error(y_validation, validation_predictions)
			if mse_min is None or mse < mse_min:
				mse_min = mse
				best_alpha = alpha
				best_l1_ratio = l1_ratio

#######################################################################
# Train the model using all the 80% data
model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=fit_intercept, max_iter=10e6)
model.fit(x_train_validation, y_train_validation)
test_predictions = model.predict(x_test).reshape((-1,1))


confusion_matrix(test_predictions, y_test, lambda y: y > 0)

title = "Elastic Net"
plot(y_test, test_predictions, title, os.path.join(cache_dir, "plot_{0}.png".format(title).replace(" ", "")))
