"""
DIFFERENT FROM V1:
- Applying grid search to optimize hyperparamters such as batch size, epochs
- Applying grid search to run a sweep of optimizer functions to find the best fit 

Will use Pima Indians dataset from UCI that describes patient's onset of diabetes (0 or 1)

Goal: Train a binary classifcation model to diagnose diabetes for Pima Indians

Each patient has following 8 attributes, and output is 9th attribute:
   1. Number of times pregnant
   2. Plasma glucose concentration 
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   Output: Class variable (0 or 1, where 1 means tested positive for diabetes)

"""

import numpy as np
from keras.models import Sequential     # for daisy chained model (each layer has 1 I/O tensor)
from keras.layers import Dense          # for densely connected NN layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


seed = 5
np.random.seed(seed)

# Load input data (CSV) as matrix of numbers, with 8 inputs and last col as output class
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input vector (X) and output class scalar (y) 
X = dataset[:,:-1]      # all features for all rows except last col
y = dataset[:,-1]        # all rows but just with last col



# Define and compile Keras models using KerasClassifier

def pimaModel_model_sweep(optimizer='adam'):
    model = Sequential();
    model.add(Dense(12, activation="relu", input_dim=8))     # 12 nodes with 8 input cols
    model.add(Dense(1,activation='sigmoid'))

    # NOTE: compile model with configurable optimzer to sweep through available choices in Keras
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def pimaModel_batch_epoch_sweep():
    model = Sequential()
    model.add(Dense(12, activation="relu", input_dim=8))     # 12 nodes in first hidden layer, first layer needs 8 since 8 inputs
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# NOTE: Use to build model for evaluating batch/epoch sizes
# model = KerasClassifier(build_fn=pimaModel_model_sweep, verbose=0)
# batch_size = [20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)

#NOTE: Use to build model for evaluating different optimizer algorithms 
model = KerasClassifier(build_fn=pimaModel_model_sweep, batch_size=20, epochs=100, verbose=0)
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizers)

# cv parameter is to determine # of folds in K-fold, n_jobs=1 is default single core operation
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X,y)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean accuracy %f with deviation (%f) with: %r" % (mean, stdev, param))