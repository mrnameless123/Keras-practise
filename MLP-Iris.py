import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
seed = 1
np.random.seed(seed)
dataframe = pandas.read_csv('iris.csv', header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#LabelEncoder from scikit-learn turn each text label
#into a vector. In this case, each of the three label are just assigned a number from 0-2
encoder = LabelEncoder()
encoder.fit(Y)
print('Label Encoder fit()', encoder.classes_)
#transform(y) : Transform labels to normalized encoding.
encoded_Y = encoder.transform(Y)

#Converts a class vector (integers) to binary class matrix.
#Argument: y: cls vector to be converted into a matrix
#num_classes: total number of classes
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    model = Sequential()
    model.add(Dense(90, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch = 800, batch_size = 5, verbose = 0)

kfold = KFold(n_splits = 10, shuffle=True, random_state=seed)
result = cross_val_score(estimator, X, dummy_y, cv = kfold)
print("Baseline: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))