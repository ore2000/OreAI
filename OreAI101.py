# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:45:53 2022

@author: Oreoluwa Adeola Babatunde

This code is to help easily get into machine learning programming without having to use things like tensorflow with huge dependencies.

Dependecnes for running class:
    1. numpy
    2. python math function
    3. python random number generator

"""
import numpy as np
import math
import random as rnd

class oreAI:
    
    def __init__(self, number_of_inputs):
        '''This code helps in the initialization of our Machine learning model'''
        
        #create a dictionary of all the weights for each layer in the neural network with biases
        self.weights_bias = {}
        
        #create a dictionary of all weights for each layer in the neural network without biases.
        self.weights = {}
        
        #initialize a variable to keep track of the number of hidde layers in the structure
        self.number_of_hidden_layers = 0
        
        #create a list to keep track of the number of nodes in each hidden layer
        self.number_of_nodes_in_each_hidden_layer = []
        
        #store the number of nputs that wil be used in the neural network
        self.num_inputs = number_of_inputs
        
        #create an array to store the inputs. We add one for the bias
        input_array = np.empty(self.num_inputs + 1, dtype = np.float32)
        
        print("creation of the input layer is done")
        
        
    def input_layer(self, number_of_nodes):
        '''Create the input layer of the machine learning model'''
        
        #Store the number of nodes for the machine learninf model
        self.num_input_nodes = number_of_nodes
        
        #get the total number of elements that will be in the weight matrix. We add +1 for the bias weight.
        number_of_vars = self.num_input_nodes*(self.num_inputs + 1)
        
        w = []
        
        #create a list of random values for the weights. currently uses values within the range of -(total number of weights) and (total number of weights)
        for i in range(number_of_vars):
            w.append(rnd.randint(-number_of_vars, number_of_vars))
            
        #create the weight matrix for the input layer:
        weight_matrix = np.reshape(w, (self.num_input_nodes,(self.num_inputs + 1)))
        
        #store the input weight matrix in the weights dictionary for the neural network.
        self.weights_bias[0] = weight_matrix
        
        #store weights without bias
        self.weights[0] = weight_matrix[:, :(weight_matrix.shape[1]-1)]
        
        
        print("Creation of the input layer matrix is done")
        
    def add_layer(self,number_of_nodes):
        '''Function to be used for creating a hidden layer'''
        
        number_of_layer_inputs = 0
        self.number_of_nodes_in_each_hidden_layer.append(number_of_nodes)
        
        #determine the number of inputs there will be for this layer.
        if self.number_of_hidden_layers == 0:
            # add 1 for the bias.
            number_of_layer_inputs = self.num_input_nodes + 1
        else:
            # add 1 for the bias.
            number_of_layer_inputs = self.number_of_nodes_in_each_hidden_layer[self.number_of_hidden_layers] + 1
            
        #get the total number of elements that will be in the weight matrix
        number_of_vars = number_of_layer_inputs*number_of_nodes
        
        w = []
        
        #create a list of random values for the weights
        for i in range(number_of_vars):
            w.append(rnd.randint(-number_of_vars, number_of_vars))
            
        #create the weight matrix for the input layer:
        weight_matrix = np.reshape(w, (number_of_nodes,number_of_layer_inputs))
        
        #increase the number of layers by one 
        self.number_of_hidden_layers = self.number_of_hidden_layers + 1
        
        #store the input weight matrix in the weights dictionary for the neural network.
        self.weights_bias[self.number_of_hidden_layers] = weight_matrix
        
        #store weights without bias
        self.weights[self.number_of_hidden_layers] = weight_matrix[:, :(weight_matrix.shape[1]-1)]
        
        print("Hidden layer has been added.")
        
        
        
    
        
if __name__=='__main__':
    ml = oreAI(3)
    ml.input_layer(6)
    ml.add_layer(3)
    ml.add_layer(3)
    ml.add_layer(3)
    print(ml.weights)
        
