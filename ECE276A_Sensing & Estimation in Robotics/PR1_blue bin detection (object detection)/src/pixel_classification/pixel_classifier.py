'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import sys
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(folder_path)

from model_setup import ModelSetup
from PerformSoftmax import perform_softmax

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    model_setting = ModelSetup()
    #self.w, self.b = model_setting.load_parameters()
    self.w = np.array([[19.37039397, -9.35861042, -9.40289135],
       [-8.6467279 , 19.65500204, -9.2323462 ],
       [-9.60991067, -9.37327879, 18.88571381]])

    self.b = np.array([0.24267098, 0.35778989, 0.17203291])
    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    out = np.matmul(X, self.w) + self.b
    prediction = perform_softmax(out)
    y = np.argmax(prediction, axis=1) + 1
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

