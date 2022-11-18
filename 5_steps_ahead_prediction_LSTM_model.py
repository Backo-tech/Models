# Importing required libraries
import pandas as pd # a library for data manipulation and analysis, with a focus on data structures and operations for manipulating numeric tables and time series
import numpy as np # library for multidimensional arrays and matrices, along with a large collection of high-level math functions for operating on arrays
from datetime import datetime, timedelta # The datetime module provides classes for manipulating dates and times
from matplotlib import pyplot as plt # plotting library
from keras.models import Sequential # a library representing an interface for working with neural networks, from the TensorFlow library
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Reading the input data
# Uloading data from an external CSV file
d = pd.read_csv('/content/****/Pinnacle.csv')

# Formating to datetime
# Date and time conversion from uploaded file to timeseries format
d['Datetime'] = [datetime.strptime(x, '%m/%d/%Y %H:%M:%S') for x in d['Datetime']]

# Plotting the variable P 
d_p = d['P']
plt.figure(figsize=(8, 5))
d_p.plot()
plt.show()

### AUXILIARY FUNCTIONS
"""
create_data_for_NN is the main function that we call, and as its name suggests, it prepares the data in the way that our network requires. Since it is an LSTM layer, it requires the data to be reshaped in a certain way.
(22736, 30, 1) (5684, 30, 1) (22736, 5) (5684, 5)
(22736, 30, 1) number of rows for the training network, 30 is the first dimension of the input sequence, 1 is the second dimension of the input sequence (i.e. it is a sequence of 30 with a height of 1)
(22736, 5) 22736 the number of tags/labels, that is, the number of lines of output sequences, 5 is the length of the output sequence that is being predicted

22736 is the number of training examples (training set)
5684 is the number of examples for validation (test set)

The function takes a variable P from the given dataframe and converts it to an array.
It then calls another function create_X_Y which converts the array into data sequences against the desired lstm architecture.
In our case it is a sequence of 30 elements, which should predict 5.
After that, this main function performs a train test split according to the given ratio.
"""

def create_X_Y(ts, lag, predict_steps) -> tuple:
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - predict_steps):
            X.append( ts[i:(i+lag)])
            Y.append( ts[(i+lag):(i+lag+predict_steps)] )

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))


    return X, Y         

def create_data_for_NN(data, Y_var, lag, predict_steps, train_test_split):
    # Extracting the main variable we want to model/forecast
    y = data[Y_var].tolist()

    # The X matrix will hold the lags of Y 
    X, Y = create_X_Y(y, lag, predict_steps)

    # Creating training and test sets 
    X_train = X
    X_test = []

    Y_train = Y
    Y_test = []

    if train_test_split > 0:
        index = round(len(X) * train_test_split)
        X_train = X[:(len(X) - index)]
        X_test = X[-index:]     
        
        Y_train = Y[:(len(X) - index)]
        Y_test = Y[-index:]

    return X_train, X_test, Y_train, Y_test

# Creating test and validation data by calling previously defined functions
Y_var = 'P'
lag = 30
predict_steps = 5
train_test_split = 0.2

X_train, X_test, Y_train, Y_test = create_data_for_NN(d, Y_var, lag, predict_steps, train_test_split)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

x_df = pd.DataFrame(X_train[:5].reshape(5,30))
print(x_df)

y_df = pd.DataFrame(Y_train[:5].reshape(5,5))
print(y_df)

Y_train[0]

# Defining the model
model = Sequential(name='lstm_5steps')
model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(lag, 1), name="lstm_0"))
model.add(LSTM(512, activation='relu', name='lstm_1'))
model.add(Dense(256, name='dense_0'))
model.add(Dense(128, name='dense_1'))
model.add(Dense(64, name='dense_2'))
model.add(Dense(32, name='dense_3'))
model.add(Dense(predict_steps, name='dense_4'))


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01)
save_checkpoint = ModelCheckpoint(save_weights_only=False, save_best_only=True, filepath='pinnacle_model_5steps_bst.h5' )

# Defining the model parameter dict 
keras_dict = {
    'x': X_train,
    'y': Y_train,
    'validation_data':[X_test, Y_test],
    'batch_size': 100,
    'epochs': 100,
    'shuffle': False,
    'callbacks':[early_stop,save_checkpoint]
}

# Fitting the model 
history = model.fit(**keras_dict)

plt.figure()
plt.plot(history.history['loss'], label='training MSE loss')
plt.plot(history.history['val_loss'], label='validation MSE loss')
plt.legend()

plt.figure()
plt.plot(history.history['mae'], label='training MAE loss')
plt.plot(history.history['val_mae'], label='validation MAE loss')
plt.legend()

# load and evaluate a saved model
from keras.models import load_model

# load model
model = load_model('/content/*****/pinnacle_model_5steps_bst.h5')
model.summary()

score = model.evaluate(X_test, Y_test)
print(score)

# Vizuelizacija modela
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



yhat = model.predict(X_test)

position = 1672
A = pd.Series(np.reshape(X_test[position], 30)).append(pd.Series(np.reshape(Y_test[position],5))).reset_index(drop=True)
B = pd.Series(np.reshape(X_test[position], 30)).append(pd.Series(np.reshape(yhat[position],5))).reset_index(drop=True)
# B = pd.Series(yhat[position])
A.index += 1 
B.index += 1 

plt.figure(figsize=(9, 5))
plt.plot(A,'b')  # blue - input sequence
plt.plot(A[30:], 'r')  # red - real data from the datset
plt.plot(B, 'b') # blue - input sequence
plt.plot(B[30:], 'g') # green - model-predicted data

model.save("AAI.h5")