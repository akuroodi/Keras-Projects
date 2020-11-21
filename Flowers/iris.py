"""

Multi-class classification model to identify different species of flowers from iris flowers UCI dataset

>95% accuracy

Inputs: 4 measurements in cm of a flower 
Output: classification of the flower as either Iris-setosa, Iris-versicolor, Iris-virginica

"""

import pandas
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# Load the dataset
dataframe = pandas.read_csv("iris.csv", header=None)        # easier to use pandas over numpy since data contains strings
dataset = dataframe.values

X = dataset[:,:-1].astype(float)
y = dataset[:, -1]

# Assigns ints to each classification
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)        

# One-hot encoding on our 3 output class variables
"""
red    blue   green
1,      0,      0
0,      1,      0
0,      0,      1
"""
dummy_y = np_utils.to_categorical(y_encoded)

# Define our NN model to pass into Keras Classifier, note nodes/hidden layers are tunable
def FlowerModel():
    model = Sequential()
    model.add(Dense(12, activation="relu", input_dim=4))     
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




