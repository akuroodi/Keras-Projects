"""
Serving as an introduction to Keras on Tensorflow in Python.

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
from keras.models import Sequential     # for linear stack of layers
from keras.layers import Dense          # for densely connected NN layers


# Load input data (CSV) as matrix of numbers, with 8 inputs and last col as output class
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input vector (X) and output class scalar (y) 
X = dataset[:,:-1]      # all features for all rows except last col
y = dataset[:,-1]        # all rows but just with last col


""" 
Define and compile a Keras model. 
Use Sequential class to add layers one at a time

For this diabetes dataset we will use 2 fully connected layers and one sigmoid output layer

"""

model = Sequential()
model.add(Dense(12, activation="relu", input_dim=8))     # 12 nodes in first hidden layer, first layer needs 8 since 8 inputs
model.add(Dense(8, activation='relu'))
#model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Adam = SGD, use cross entropy as common cost func for binary classification and report accuracy of said classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X,y) # evaluate function returns 2 value list, first element is loss which we discard

print('Accuracy for classifying diabetes is: %.2f ' % (accuracy*100) )

