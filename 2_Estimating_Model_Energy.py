# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:44:05 2024

@author: User
"""

import warnings

import numpy as np
import tensorflow as tf

import keras_spiking

warnings.simplefilter("ignore")
tf.get_logger().addFilter(lambda rec: "Tracing is expensive" not in rec.msg)


#Using ModelEnergy
# build an example model
inp = x = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(filters=2, kernel_size=(7, 7))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=128)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(units=10)(x)

model = tf.keras.Model(inp, x)
model.summary()


# estimate model energy
energy = keras_spiking.ModelEnergy(model)
energy.summary(print_warnings=False)


energy.summary(
    columns=(
        "name",
        "energy cpu",
        "energy gpu",
        "synop_energy cpu",
        "synop_energy gpu",
        "neuron_energy cpu",
        "neuron_energy gpu",
    ),
    print_warnings=False,
)



energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 28, 28)))
energy.summary(
    columns=(
        "name",
        "rate",
        "synop_energy cpu",
        "synop_energy loihi",
        "neuron_energy cpu",
        "neuron_energy loihi",
    ),
    print_warnings=False,
)


energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 28, 28, 1)) * 5)
energy.summary(
    columns=(
        "name",
        "rate",
        "synop_energy cpu",
        "synop_energy loihi",
        "neuron_energy cpu",
        "neuron_energy loihi",
    ),
    print_warnings=False,
)




#Adding custom devices

keras_spiking.ModelEnergy.register_device(
    "my-gpu", energy_per_synop=1e-9, energy_per_neuron=2e-9, spiking=False
)
energy.summary(columns=("name", "energy gpu", "energy my-gpu"), print_warnings=False)



#Temporal processing

energy.summary(
    columns=("name", "energy cpu", "energy loihi"),
    timesteps_per_inference=10,
    print_warnings=False,
)

energy.summary(
    columns=("name", "energy cpu", "energy loihi"),
    timesteps_per_inference=20,
    print_warnings=False,
)

print("dt=0.001 Begins \n\n")

energy.summary(
    columns=("name", "energy cpu", "energy loihi"), dt=0.001, print_warnings=False
)

print("dt=0.001 Ends \n")


print("dt=0.002 Begins \n\n")
energy.summary(
    columns=("name", "energy cpu", "energy loihi"), dt=0.002, print_warnings=False
)
print("dt=0.002 Ends \n")



# add a new input dimension (None) representing
# temporal data of unknown length
inp = x = tf.keras.Input((None, 28, 28, 1))
# the TimeDistributed wrapper can be used to apply
# non-temporal layers to temporal inputs
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(filters=2, kernel_size=(7, 7))
)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
# some layers, like Dense, can operate on temporal data
# without requiring a TimeDistributed wrapper
x = tf.keras.layers.Dense(units=128)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(units=10)(x)

temporal_model = tf.keras.Model(inp, x)
temporal_model.summary()




#If we compare the energy estimates of the temporal and non-temporal models 
#we can see that they are the same, because KerasSpiking is automatically 
#assuming that the non-temporal model will be translated into a temporal model:
energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 28, 28, 1)))
energy.summary(
    columns=("name", "energy cpu", "energy loihi"),
    timesteps_per_inference=10,
    print_warnings=False,
)


# note that we add a temporal dimension to our example data (which does not need to be
# the same length as timesteps_per_inference)
energy = keras_spiking.ModelEnergy(
    temporal_model, example_data=np.ones((32, 5, 28, 28, 1))
)
energy.summary(
    columns=("name", "energy cpu", "energy loihi"),
    timesteps_per_inference=10,
    print_warnings=False,
)



#in some cases the Keras model definition can be ambiguous as to 
#whether it represents a temporal or non-temporal model.

inp = tf.keras.Input((28, 28))
x = tf.keras.layers.ReLU()(inp)
model = tf.keras.Model(inp, x)
model.summary()


energy = keras_spiking.ModelEnergy(model)
energy.summary(
    columns=("name", "output_shape", "neurons", "energy cpu"), print_warnings=False
)



#You can signal to ModelEnergy that the ReLU layer should be 
#considered temporal by wrapping it in a TimeDistributed layer:

inp = tf.keras.Input((28, 28))
x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(inp)
model = tf.keras.Model(inp, x)

energy = keras_spiking.ModelEnergy(model)
energy.summary(
    columns=("name", "output_shape", "neurons", "energy cpu"), print_warnings=False
)


#we could have changed the shape of the first dimension to None, in which case ModelEnergy 
#will assume that that dimension represents time, without the need for a TimeDistributed wrapper.
inp = tf.keras.Input((None, 28))
x = tf.keras.layers.ReLU()(inp)
model = tf.keras.Model(inp, x)

energy = keras_spiking.ModelEnergy(model)
energy.summary(
    columns=("name", "output_shape", "neurons", "energy cpu"), print_warnings=False
)




#Using SpikingActivation layers

#You may have noticed above that we have been silencing some warnings.
#Let’s see what those warnings are:
inp = tf.keras.Input((None, 32))
x = tf.keras.layers.Dense(units=64)(inp)
x = tf.keras.layers.ReLU()(x)
model = tf.keras.Model(inp, x)

energy = keras_spiking.ModelEnergy(model, example_data=np.ones((8, 10, 32)))
energy.summary(columns=("name", "output_shape", "energy loihi"), print_warnings=True)


#n order to provide a useful estimate for spiking devices, we assume that any non-spiking neurons
# will be converted to spiking neurons when the model is mapped to the device
inp = tf.keras.Input((None, 32))
x = tf.keras.layers.Dense(units=64)(inp)
x = keras_spiking.SpikingActivation("relu")(x)
model = tf.keras.Model(inp, x)

energy = keras_spiking.ModelEnergy(model, example_data=np.ones((8, 10, 32)))
energy.summary(columns=("name", "output_shape", "energy loihi"))  



#Deploying to real devices
#we deploy using Nengo

# pylint: disable=wrong-import-order

import nengo_dl  #Leo: I tried to pip install nengo_dl but it didn't work
import nengo_loihi

converter = nengo_dl.Converter(model, temporal_model=True, inference_only=True)

#The advantage of the Nengo ecosystem is that once we have a Nengo model, 
#we can run that model on any Nengo-supported hardware platform. 
with nengo_loihi.Simulator(converter.net) as sim:
    sim.run_steps(10)

print(sim.data[converter.outputs[model.output]].shape)






    