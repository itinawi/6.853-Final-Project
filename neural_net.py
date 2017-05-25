import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys


# the python script will take in a number as an argument, and the number corresponds
#  to which data file the script will run on. The options are:
#			1 - old_data.csv
#			2 - exported_rounds.csv

assert(len(sys.argv) > 0)
print(sys.argv[1])
file_number = int(sys.argv[1])
file_name = ""
input_parameters = 0
if file_number == 1:
	file_name = "old_data.csv"
	input_parameters = 13
elif file_number == 2:
	file_name = "exported_rounds.csv"
	input_parameters = 27


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print(file_name)
# load dataset
dataframe = pd.read_csv(file_name, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables

# print(dataset)

# usr_strat is the user's strategy column, ie. the labels we are trying to predict
start_index = 0
if file_number == 1:
	start_index = 3
	usr_strat_index = 1
	opp_strat = dataset[1:,2].astype(float)
elif file_number == 2:
	start_index = 2
	usr_strat_index = 0
X = dataset[1:,start_index:].astype(float)
usr_strat = dataset[1:,usr_strat_index].astype(float)
print(usr_strat)
print(np.shape(X))
# X = np.concatenate(X1, X2)
# Y = dataset[:,60]



# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=input_parameters, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

## uncomment to run the baseline model
## evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, usr_strat, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=input_parameters, kernel_initializer='normal', activation='relu'))
	model.add(Dense(30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

'''
## uncomment to run the larger model
print("Running the large model...")
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=3, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, usr_strat, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

