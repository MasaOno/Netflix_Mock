'''Code for training models'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils.input_fn_utils import build_parsing_serving_input_fn
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import sys

import evaluate
import models
import utils

tf.logging.set_verbosity(tf.logging.ERROR)

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta'
VALIDATION_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'
EPOCHS = 100
BATCH_SIZE = 64
BEST_ACCURACY = 0.0

CONFIG = tf.contrib.learn.RunConfig(
    		save_checkpoints_steps = 1000,
    		keep_checkpoint_max = 1,
    		keep_checkpoint_every_n_hours = sys.maxint, 
)

def train_input_fn(dir):
	'''Returns training input function'''
	features, labels = utils.get_training(dir)
	return features, labels

def create_callable_train_input_fn(dir):
	'''Returns callable training input function for predictions'''
	data_dir = dir
	def callable_input():
		return train_input_fn(data_dir)
	return callable_input

def get_feature_column(features):
	'''Returns infered feature column from features'''
	feature_column = tf.contrib.learn.infer_real_valued_columns_from_input(features)
	return feature_column

def get_estimator(params, model_dir, feature_column, CONFIG):
	'''Returns estimator object'''
	return models.DNN_regressor(params, model_dir, feature_column, CONFIG)

def train(params, hyperparam_search=False, dimensions=None, n_calls=None):
	'''Start training'''
	print 'Loading data...'
	model_dir = './models/nn/'
	x, y = train_input_fn(TRAIN_DATA_MINI)
	feature_column = get_feature_column(x)
	print 'Training...'
	#=================
	if hyperparam_search:
		lst = [params['layers'], params['units'], params['optimizer'], params['learning_rate'], params['activation_fn'], params['dropout']]
		@use_named_args(dimensions=dimensions)
		def search(layers, units, optimizer, learning_rate, activation_fn, dropout):
			params['layers'] = layers
			params['units'] = units
			params['optimizer'] = optimizer
			params['learning_rate'] = learning_rate
			params['activation_fn'] = activation_fn
			params['dropout'] = dropout
			print params
			new_model_dir = utils.create_model_dir(model_dir, params)
			estimator = get_estimator(params, new_model_dir, feature_column, CONFIG)
			val_x, val_y = train_input_fn(VALIDATION_DATA_MINI)
			validation_monitor = utils.get_validation_monitor(val_x, val_y)
			history = estimator.fit(x = x,
								    y = y,
								    steps = EPOCHS,
							   	    batch_size = BATCH_SIZE,
								    monitors = [validation_monitor])
			validation_input = create_callable_train_input_fn(VALIDATION_DATA_MINI)
			accuracy = evaluate.get_accuracy_model(estimator, validation_input)
			global BEST_ACCURACY
			print 'Accuracy: ' + str(accuracy)
			if accuracy > BEST_ACCURACY:
				print 'Exporting model...'
				feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
				serving_input_fn = build_parsing_serving_input_fn(feature_spec)
				estimator.export_savedmodel(new_model_dir, serving_input_fn)
				BEST_ACCURACY = accuracy
			del estimator
			tf.reset_default_graph()
			return -accuracy
		result = utils.run_hyperparam_search(search, dimensions, n_calls, lst)
		global BEST_ACCURACY
		return BEST_ACCURACY
	#=================
	else:
		print params
		estimator = get_estimator(params, model_dir, feature_column, CONFIG)
		val_x, val_y = train_input_fn(VALIDATION_DATA_MINI)
		validation_monitor = utils.get_validation_monitor(val_x, val_y)
		history = estimator.fit(x = x,
							    y = y,
					 		    steps = EPOCHS,
					 		    batch_size = BATCH_SIZE,
					 		    monitors = [validation_monitor])
		validation_input = create_callable_train_input_fn(VALIDATION_DATA_MINI)
		return evaluate.get_accuracy_model(estimator, validation_input)

if __name__ == "__main__":
	print 'Training nn model...'
	params = {'layers': 2,
			  'units': 128,
			  'n_classes': 6,
			  'optimizer': 'adam',
			  'learning_rate': 0.01,
			  'activation_fn': 'relu',
			  'dropout': 0.1}
	dimensions = [Integer(low=1, high=5, name='layers'),
				  Integer(low=100, high=300, name='units'),
				  Categorical(categories=['sgd', 'adagrad', 'momentum', 'adam', 'rms'], name='optimizer'),
				  Real(low=0.0001, high=0.1, name='learning_rate'),
				  Categorical(categories=['relu', 'sigmoid', 'softmax'], name='activation_fn'),
				  Real(low=0.0, high=0.5, name='dropout')]
	result = train(params, hyperparam_search=True, dimensions=dimensions, n_calls=50)
	print 'Final accuracy: ' + str(result)

