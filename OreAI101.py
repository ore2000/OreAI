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
        
        #create a dictionary of all the weights for each layer in the neural network
        self.weights = {}
        
        #store the number of nputs that wil be used in the neural network
        self.num_inputs = number_of_inputs
        
        #create an array to store the inputs
        input_array = np.empty(self.num_inputs, dtype = np.float32)
        
        print("creation of the input layer is done")
        
    def input_layer(self, number_of_nodes):
        '''Create the input layer of the machine learning model'''
        #Store the number of nodes for the machine learninf model
        self.num_input_nodes = number_of_nodes
        
        #get the total number of elements that will be in the weight matrix
        number_of_vars = self.num_input_nodes*self.num_inputs
        
        w1 = []
        
        #create a list of random values for the weights
        for i in range(number_of_vars):
            w1.append(rnd.randint(-number_of_vars, number_of_vars))
            
        #create the weight matrix for the input layer:
        weight_matrix = np.reshape(w1, (self.num_inputs,self.num_input_nodes))
        
        #store the input weight matrix in the weights dictionary for the neural network.
        self.weights[0] = weight_matrix
        print("Creation of the input layer matrix is done")
    
        
if __name__=='__main__':
    ml = oreAI(3)
    ml.input_layer(6)
        
