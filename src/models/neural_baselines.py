"""
=================================
Baseline Model Creation Utilities 
=================================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

Contains the routines to define the CNN and LSTM baseline models as described in the reference paper. 
Makes use of the Tensorflow framework. 

**Functions**
    :function create_neural_model: returns the compiled model asked for in input

"""
from tensorflow import keras
import tensorflow as tf
# import tensorflow.keras as keras

#pylint: disable=import-error
from tensorflow.keras import layers, metrics
from tensorflow.keras.callbacks import EarlyStopping


def create_neural_model(model_type, n_timesteps, n_features):
    if model_type == 'cnn':
        model = keras.Sequential()
        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding = 'same' ,input_shape=(n_timesteps,n_features)))
        model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding = 'same'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding = 'same'))
        model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.5))
        #pooling
        #dropout
        #batch training
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

        return model
    
    elif model_type == 'lstm':
        model = keras.Sequential()
        model.add(layers.LSTM(128, input_shape=(n_timesteps, n_features)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

        return model
    