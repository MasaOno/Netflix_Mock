# Utils
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.contrib import predictor

'''Code to blend and evaluate trained models'''

def get_accuracy_model(model, data, tfmodel = True):
	if tfmodel: 
		_, labels = data[:, 0]
		#predictions = np.zeros(len(labels))

		# STILL NOT SURE HOW DATA IS PASSED BECAUSE IN THE OLD VERSION IT LOOKED LIKE DATA WAS A FUNCTION!??!?!?
		predictions = model.predict(input_fn = data)
		
		# convert predictions to 0/1
		super_threshold_indices = predictions >= 0.5
		predictions[super_threshold_indices] = 1

		below_threshold_indices = predictions < 0.5
		predictions[below_threshold_indices] = 0

	 	score = 0
	 	for j, p in enumerate(predictions): 
	 		if p == labels[j]:
	 			score += 1

	 	return float(score) / len(predictions)
	else: 
		return 5

def get_accuracy_dir(modelfile, data, tfmodel = True):
	# modelfile = string
	# data = np array of data with labels in 1st column
	if tfmodel: 

		# need to take model out of modelfile
		predict_fn = predictor.from_saved_model(modelfile)


		labels = data[:, 0]
		x = data[:,range(1, len(data))]
		#predictions = np.zeros(len(labels))
		#prediction = model.predict(input_fn = data[i])
		
		# not sure if this how you give data to it
		predictions = predict_fn({"x": x,"y":labels})
		# and not sure what comes out

		# convert predictions to 0/1
		# super_threshold_indices = prediction >= 0.5
		# prediction[super_threshold_indices] = 1

		# below_threshold_indices = prediction < 0.5
		# prediction[below_threshold_indices] = 0


	 	score = 0
	 	for j, p in enumerate(predictions): 
	 		if p == labels[j]:
	 			score += 1

	 	return float(score) / len(predictions)

	else: 
		return 5

def ensemble(modelfolder, testdata):
	
	return 5#weights, accuracy


def predict(weights, testdata):
	# create text file with predictions
	return 5


