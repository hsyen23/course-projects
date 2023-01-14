# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:05:20 2022

@author: Yen
"""
import sys
import os
folder_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(folder_path)
from os.path import exists
import numpy as np
from generate_rgb_data import read_pixels
import math
from PerformSoftmax import perform_softmax

class ModelSetup():
    
    def load_parameters(self, path = None):
        '''
        Load paramteres for model.
        If path isn't provided, then it will search
        in current direcotry.

        Parameters
        ----------
        path : String, optional
            Absolute path of directory which contains "model_parameters_w.txt" and file "model_parameters_b.txt". 
            The default is None.

        Returns
        -------
        w : Numpy.ndarray
            Parameter vector (weight)
        b : Numpy.ndarray
            Parameter vector (bias)
        '''
        if path == None:
            w_path = os.path.join(folder_path, 'model_parameters_w.txt')
            b_path = os.path.join(folder_path, 'model_parameters_b.txt')
            assert exists(w_path)
            assert exists(b_path)
            w = np.loadtxt(w_path, dtype=float)
            b = np.loadtxt(b_path, dtype=float)
            return w, b
        else:
            w_path = path + '/model_parameters_w.txt'
            b_path = path + '/model_parameters_b.txt'
            assert exists(w_path)
            assert exists(b_path)
            w = np.loadtxt(w_path, dtype=float)
            b = np.loadtxt(b_path, dtype=float)
            return w, b
        
    def create_parameters(self, input_num, output_num):
        '''
        Create weights and bias for model based on provided information.
        File will generate at current directory with name 
        "model_parameters_w.txt" and "model_parameters_b.txt".
        
        Parameters
        ----------
        input_num : Int
        Number for input.
        output_num : Int
        Number for output.

        Returns
        -------
        None.
        
        '''
        w = np.random.randn(input_num, output_num)
        b = np.random.randn(output_num)
        np.savetxt('model_parameters_w.txt', w)
        np.savetxt('model_parameters_b.txt', b)
    
    def save_parameters(self, w, b, path = None):
        '''
        Write w and b into "model_parameters_w.txt" and "model_parameters_b.txt".
        If path isn't provided then searching in current directory.

        Parameters
        ----------
        w : Numpy.ndarray
            Parameter vector (weight)
        b : Numpy.ndarray
            Parameter vector (bias)
        path : String, optional
            Absolute path of directory which contains "model_parameters_w.txt" and file "model_parameters_b.txt". 
            The default is None.

        Returns
        -------
        None.

        '''
        if path == None:
            np.savetxt('model_parameters_w.txt', w)
            np.savetxt('model_parameters_b.txt', b)
        else:
            w_path = path + '/model_parameters_w.txt'
            b_path = path + '/model_parameters_b.txt'
            np.savetxt(w_path, w)
            np.savetxt(b_path, b)
            
    def get_trainingData(self):
        '''
        Gather all training data under the relative path "data/training".
        Return a structure that works on my model.

        Returns
        -------
        data_value : Numpy.ndarray
            n * 3 array, nth row presents a data
        label_value : Numpy.ndarray
            n * 3 array, nth row presents corresponding label for nth data
            [1,0,0] for red, [0,1,0] for green, [0,0,1] for blue

        '''
        # get R part
        data_value = read_pixels('training_color/red')
        l = len(data_value)
        label_value = np.zeros([l, 8])
        label_value[:, 0] = 1
        # get G part
        g_part = read_pixels('training_color/green')
        l = len(g_part)
        g_label = np.zeros([l, 8])
        g_label[:, 1] = 1
        data_value = np.concatenate((data_value, g_part), axis=0)
        label_value = np.concatenate((label_value, g_label), axis=0)
        # get B part
        b_part = read_pixels('training_color/blue')
        l = len(b_part)
        b_label = np.zeros([l, 8])
        b_label[:, 2] = 1
        data_value = np.concatenate((data_value, b_part), axis=0)
        label_value = np.concatenate((label_value, b_label), axis=0)
        #
        # get Skyblue part
        skyblue_part = read_pixels('training_color/skyblue')
        l = len(skyblue_part)
        skyblue_label = np.zeros([l, 8])
        skyblue_label[:, 3] = 1
        data_value = np.concatenate((data_value, skyblue_part), axis=0)
        label_value = np.concatenate((label_value, skyblue_label), axis=0)
        #
        # get Black part
        black_part = read_pixels('training_color/black')
        l = len(black_part)
        black_label = np.zeros([l, 8])
        black_label[:, 4] = 1
        data_value = np.concatenate((data_value, black_part), axis=0)
        label_value = np.concatenate((label_value, black_label), axis=0)
        #
        # get White part
        white_part = read_pixels('training_color/white')
        l = len(white_part)
        white_label = np.zeros([l, 8])
        white_label[:, 5] = 1
        data_value = np.concatenate((data_value, white_part), axis=0)
        label_value = np.concatenate((label_value, white_label), axis=0)
        #
        # get Yellow part
        yellow_part = read_pixels('training_color/yellow')
        l = len(yellow_part)
        yellow_label = np.zeros([l, 8])
        yellow_label[:, 6] = 1
        data_value = np.concatenate((data_value, yellow_part), axis=0)
        label_value = np.concatenate((label_value, yellow_label), axis=0)
        #
        # get Gray part
        gray_part = read_pixels('training_color/gray')
        l = len(gray_part)
        gray_label = np.zeros([l, 8])
        gray_label[:, 7] = 1
        data_value = np.concatenate((data_value, gray_part), axis=0)
        label_value = np.concatenate((label_value, gray_label), axis=0)
        #
        return data_value, label_value
    
    def shuffle_data(self, data, label):
        '''
        Shuffle data and label.
        The relationship of data and corresponding label won't chagne.

        Parameters
        ----------
        data : Numpy.ndarray
            n * 3 array, nth row presents a data
        label : Numpy.ndarray
            n * 3 array, nth row presents corresponding label for nth data
            [1,0,0] for red, [0,1,0] for green, [0,0,1] for blue

        Returns
        -------
        shuffled_data : Numpy.ndarray
            same as input, but order is shuffled.
        shuffled_label : Numpy.ndarray
            Label vector corresponding to data.

        '''
        l = len(data)
        ref = np.random.permutation(l)
        shuffled_data = np.zeros([np.shape(data)[0], np.shape(data)[1]])
        shuffled_label = np.zeros([np.shape(label)[0], np.shape(label)[1]])
        for i in range(l):
            index = ref[i]
            shuffled_data[i] = data[index]
            shuffled_label[i] = label[index]
        return shuffled_data, shuffled_label
    
    def train_model(self, w, b, dataset, lr, num_iter):
        '''
        Based on provided dataset to improve parameters(w, b) with hyperparamters
        learning rate and number of iteration.

        Parameters
        ----------
        w : Numpy.ndarray
            Parameter vector (weight)
        b : Numpy.ndarray
            Parameter vector (bias)
        dataset : List[data, label]
            A list contains data and label.
            data is n * 3 array, nth row presents a data.
            label is n * 3 array, nth row presents corresponding label for nth data.
        lr : Float
            Learning rate.
        num_iter : Int
            Number of iteration for this dataset.

        Returns
        -------
        w : Numpy.ndarray
            Updated parameter vector (weight).
        b : Numpy.ndarray
            Updated parameter vector (bias).

        '''
        data_value, label_value = dataset
        # data_value is n * 3 ex:[0.12, 0.23, 0.45]
        # lable_value is n * 3 ex:[0, 0, 1]
        num_input = np.shape(w)[0]
        num_output = np.shape(w)[1]
        num_data = len(label_value)
        # iteration
        for i in range(num_iter):
            # first, need to create softmax result table
            v = np.matmul(data_value, w) + b
            v = perform_softmax(v)
            
            # calculate derivative and update parameter
            d_w = np.zeros([num_input, num_output])
            d_b = np.zeros([num_output])
            for data_index in range(num_data):
                # calculate loss for one data
                for col_index in range(num_output):
                    d_w[:, col_index] = d_w[:, col_index] + (-label_value[data_index][col_index] + v[data_index][col_index]) * data_value[data_index]
                    d_b[col_index] = d_b[col_index] -label_value[data_index][col_index] + v[data_index][col_index]
            d_w = d_w / num_data
            d_b = d_b / num_data
            
            w = w - lr * d_w
            b = b - lr * d_b
            '''
            #print loss in specified iteration
            if (i+1) % 10 == 0:
                # using cross-entropy calculate loss (later move it to when i == multiplication of 10 then display loss)
                acc_for_all = 0
                for data_index in range(num_data):
                    acc_for_one_row = 0
                    for col_index in range(num_output):
                        acc_for_one_row += label_value[data_index][col_index] * math.log(v[data_index][col_index]) + (1 - label_value[data_index][col_index]) * math.log(1 - v[data_index][col_index])
                        acc_for_one_row = - acc_for_one_row
                    acc_for_all += acc_for_one_row
                
                loss = acc_for_all / num_data
                print('{}th iteration, Loss: {}'.format(i+1, loss))
            '''
        # print loss after this batch completes training
        acc_for_all = 0
        for data_index in range(num_data):
            acc_for_one_row = 0
            for col_index in range(num_output):
                acc_for_one_row += label_value[data_index][col_index] * math.log(v[data_index][col_index]) + (1 - label_value[data_index][col_index]) * math.log(1 - v[data_index][col_index])
                acc_for_one_row = - acc_for_one_row
                acc_for_all += acc_for_one_row
                    
        loss = acc_for_all / num_data
        print('Loss : {}'.format(loss))
        return w, b