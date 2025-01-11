import numpy as np


import os
os.environ["KERAS_BACKEND"] = "torch"
import keras



# Generate example data
num_samples = 400  # Number of data samples
xTrain = np.random.random((num_samples, 20*6))  # Random input data
yTrain = np.random.random((num_samples, 30))  # Random target data

# Define the input shape and model
inputShape = (20*6,)

if False:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=inputShape),
            keras.layers.Dense(30, activation="elu"),
        ]
    )


elif True:
    dimModel = 50 # dimensionality of model
    dimModel2 = 30
    
    input1 = keras.layers.Input(shape=inputShape)
    
    layerA = keras.layers.Dense(dimModel, activation='elu')(input1)

    layerB = keras.layers.Dense(dimModel, activation='elu')(input1)

    mul0 = keras.layers.Multiply()([layerA, layerB])
    
    layerC = keras.layers.Dense(dimModel2, activation=None)(mul0)

    model = keras.models.Model(inputs=input1, outputs=layerC)
    

# Compile the model
model.compile(
    optimizer="adamW",  # Optimizer
    loss="mse",        # Mean squared error for regression tasks
    metrics=["mae"],   # Mean absolute error as a metric
)


batchSize = 16
epochs = 0

callbacks = [
    #keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    ###keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    xTrain,
    yTrain,
    batch_size=batchSize,
    epochs=epochs,
    validation_split=0.02,
    callbacks=callbacks,
)




    
if False:
    # https://keras.io/api/layers/
    
    pass



    
from math import cos
import random

class RngB(object):
    def __init__(self):
        self.z = 0.01
       
    def nextReal(self):
        z0 = pow(self.z, 1.0018383)
        z1 = cos(z0*1.928718927372e5)
        self.z += 1.0
        return (z1+1.0) * 0.5


def genRngVec(size, rng):
    z = []
    for z1 in range(10):
        z.append(rng.nextReal()*2.0-1.0)
    return z

if True:
    dimSize = 20
    countVecs = 50
    
    rng = RngB()
    
    vecs = []
    for z1 in range(countVecs):
        vecs.append( genRngVec(dimSize, rng) )
    
    for iVec in vecs:
        print(iVec)
    
    
    
    

    tokens = [0, 7, 7, 4, 0, 2]



    
    seqMaxLen = 6
    
    # insert spacing tokens to teach MLP about sequences
    for it0 in range(seqMaxLen-2):
        
        tokens = tokens[:]
        
        idxRandom = random.randint(0, len(tokens)-1)
        tokens.insert(idxRandom, 0) # Insert 0 at the random index
        
        tokens = tokens[-seqMaxLen:]

    print(tokens)




    # convert tokens to real valued vector
        
    arr = []
    for itToken in tokens:
        arr = arr + vecs[itToken]
    
    print(arr)
    
