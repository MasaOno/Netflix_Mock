'''Code for training nn'''
import numpy as np
import tensorflow as tf

import evaluate
import models
import utils

tf.logging.set_verbosity(tf.logging.ERROR)

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta'
VALIDATION_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'
EPOCHS = 50000
BATCH_SIZE = 64

def train_input_fn(dir):
	'''Returns training input function'''
	features, labels, _ = utils.get_training(dir)
	return features, labels

def create_callable_train_input_fn(dir):
	'''Returns callable training input function for predictions'''
	data_dir = dir
	def callable_input():
		return train_input_fn(data_dir)
	return callable_input

def get_estimator(params):
	'''Returns estimator object'''
	return models.tensor_forest(params)

def train(params):
	'''Start training'''
	print 'Loading data...'
	model_dir = './models/tf/'
	x, y = train_input_fn(TRAIN_DATA_MINI)
	print 'Training...'
	#=================
	estimator = get_estimator(params)
	val_x, val_y = train_input_fn(VALIDATION_DATA_MINI)
	validation_monitor = utils.get_validation_monitor(val_x, val_y)
	history = estimator.fit(x = x,
							y = y,
					 		steps = EPOCHS,
					 		batch_size = BATCH_SIZE,
					 		monitors = [validation_monitor])
	validation_input = create_callable_train_input_fn(VALIDATION_DATA_MINI)
	return evaluate.get_rmse_model(estimator, validation_input)

if __name__ == '__main__':
	print 'Training tf model...'
	params = {'num_classes': 1,
			  'num_features': 3,
			  'regression': '1',
			  'num_trees': 20,
			  'max_nodes': '20'}
	result = train(params)
	print 'Final rmse: ' + str(result)