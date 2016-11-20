import numpy
import pandas
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("credit_card_transactions.csv", header=0)
dataset = dataframe.values
X = dataset[:,0:30]
Y = dataset[:,30]

model = Sequential()
model.add(Dense(30, input_dim=30, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X, Y, validation_split=0.33, nb_epoch=10, batch_size=2, callbacks=callbacks_list, verbose=1)
