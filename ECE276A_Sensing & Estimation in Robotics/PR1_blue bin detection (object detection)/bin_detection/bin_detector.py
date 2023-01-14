'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import sys
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(folder_path)

import numpy as np
import cv2
from skimage.measure import label, regionprops
from model_setup import ModelSetup
from PerformSoftmax import perform_softmax
import skimage.morphology
class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		# load parameters
		model_setting = ModelSetup()
		self.w, self.b = model_setting.load_parameters()
		   
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		# create a blank image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		h = img.shape[0]
		w = img.shape[1]
		mask_img = np.zeros([h, w])
		# loop each pixel in img
		for i in range(h):
			for j in range(w):
				x = [img[i, j, :] / 255]
				out = np.matmul(x, self.w) + self.b
				prediction = perform_softmax(out)
				y = np.argmax(prediction, axis=1) + 1
				# draw on mask
				if y == 3:
					mask_img[i][j] = 1
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		img_w = img.shape[1]
		img_h = img.shape[0]
		
		left_right_footprint = np.array([[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0]])
		#up_down_footprint = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

		erosion_1 = skimage.morphology.binary_erosion(img,left_right_footprint)
		erosion_1 = erosion_1.astype(float)
		
		erosion_2 = skimage.morphology.binary_erosion(erosion_1,left_right_footprint)
		erosion_2 = erosion_2.astype(float)
		
		#dialation_1 = skimage.morphology.binary_dilation(erosion_2,up_down_footprint)
		#dialation_1 = dialation_1.astype(float)
		
		finaloutput = skimage.morphology.label(erosion_2, connectivity=2)
		props = regionprops(finaloutput)
		
		another_mask = np.zeros([img_h, img_w])
		# visualize box
		for obj in props:
		#see all
			y1 = obj['bbox'][0]
			x1 = obj['bbox'][1]
			y2 = obj['bbox'][2]
			x2 = obj['bbox'][3]
			if (x2-x1) < 0.05 * img_w or (x2-x1) > 0.8 * img_w:
				continue

			# fill another mask
			another_mask[y1:y2, x1:x2] = 1
			
		boxes = []
		another_mask = skimage.morphology.label(another_mask, connectivity=1)
		props = regionprops(another_mask)
		for obj in props:
			y1 = obj['bbox'][0]
			x1 = obj['bbox'][1]
			y2 = obj['bbox'][2]
			x2 = obj['bbox'][3]
			if (x2 - x1) >= (y2-y1):
				continue
			boxes.append([x1,y1,x2,y2])
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


