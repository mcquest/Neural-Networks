# Import file libraries for math
import numpy as np
import tensorflow as tf
from tensorflow import keras

#function OR
def ORAI():

    # Rules for "or" ||
    X1 = [[0,0], [1,0], [0,1], [1,1]] # input
    y1 = [0,1,1,1] # output Xor = [0,1,1,0]

    model = keras.models.Sequential()
    # add Input shape "2," ~ array of length 2
    model.add(keras.Input(shape = (2,)))
    # create perceptron - single neuron to allow calculation
    model.add(keras.layers.Dense(units = 1, activation = "sigmoid"))
    #print(model.summary)

    """
    perceptrons have weight 
    that is multiplied by input 
    then added to bias
    """
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = keras.losses.MeanSquaredError())

    model.fit(np.array(X1), np.array(y1), batch_size = 4, epochs = 5000)

    varq = model.predict(np.array([[1, 0], [1, 1]]))

    print(round(varq[0][0]))
    print(round(varq[1][0]))

    #Hello World
    #print(tf.reduce_sum(tf.random.normal([1000,1000])))

    
# function exclusively-or
def XORAI ():

    # Rules for "or" ||
    X2 = [[0,0], [1,0], [0,1], [1,1]] # input
    y2 = [0,1,1,0] # output Xor = [0,1,1,0]

    model = keras.models.Sequential()
    # add Input shape "2," ~ array of length 2
    model.add(keras.Input(shape = (2,)))
    # create perceptron - single neuron to allow calculation - layer
    # 8 is optimal
    model.add(keras.layers.Dense(units = 3, activation = "sigmoid"))
    ## add another layer here
    model.add(keras.layers.Dense(units = 1, activation = "sigmoid"))
    #print(model.summary)

    """
    perceptrons have weight 
    that is multiplied by input 
    then added to bias
    """

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = keras.losses.MeanSquaredError())

    model.fit(np.array(X2), np.array(y2), batch_size = 4, epochs = 500)

    varq = model.predict(np.array([[1, 0], [1, 1]]))

    print(round(varq[0][0]))
    print(round(varq[1][0]))

#XORAI()


#What to Ai next
def ANDAI ():

    # Rules for "or" ||
    X3 = [[0,0], [1,0], [0,1], [1,1]] # input
    y3 = [0,0,0,1] # output Xor = [0,1,1,0]

    model = keras.models.Sequential()
    # add Input shape "2," ~ array of length 2
    model.add(keras.Input(shape = (2,)))
    # create perceptron - single neuron to allow calculation - layer
    # 8 is optimal
    #model.add(keras.layers.Dense(units = 3, activation = "sigmoid"))
    ## add another layer here
    model.add(keras.layers.Dense(units = 1, activation = "sigmoid"))
    #print(model.summary)

    """
    perceptrons have weight 
    that is multiplied by input 
    then added to bias
    """

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = keras.losses.MeanSquaredError())

    model.fit(np.array(X3), np.array(y3), batch_size = 4, epochs = 500)

    varq = model.predict(np.array([[1, 0], [1, 1]]))

    print(round(varq[0][0]))
    print(round(varq[1][0]))

ANDAI()

# Tic Tac Toe
