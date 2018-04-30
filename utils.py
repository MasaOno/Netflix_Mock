'''Code for utilities'''
import tensorflow as tf
import numpy as np
from skopt import gp_minimize

def get_param(param, key):
	'''Returns param if key exists'''
	if key in param:
		return param[key]
	return None

def get_activation(activation):
	'''Returns activations'''
	if activation == 'relu':
		return tf.nn.relu
	elif activation == 'tanh':
		return tf.tanh
	elif activation == 'sigmoid':
		return tf.sigmoid
	elif activation == 'softmax':
		return tf.nn.softmax
	return None

def get_optimizer(optimizer, learning_rate):
	'''Returns optimizer'''
	if optimizer == 'sgd':
		return tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
	elif optimizer == 'adagrad':
		return tf.train.AdagradOptimizer(learning_rate = learning_rate)
	elif optimizer == 'momentum':
		return tf.train.MomentumOptimizer(learning_rate = learning_rate,
										  momentum = 0.1) # change later
	elif optimizer == 'adam':
		return tf.train.AdamOptimizer(learning_rate = learning_rate)
	elif optimizer == 'rms':
		return tf.train.RMSPropOptimizer(learning_rate = learning_rate)
	return None

def dummy():
	data_dir = '/Users/masaono/Desktop/cs156b/um/train_all.dta'
	num_lines = sum(1 for line in open(data_dir))
	curr_num = 0
	with open(data_dir) as f:
		b = open('/Users/masaono/Desktop/cs156b/um/train.dta', 'w')
		c = open('/Users/masaono/Desktop/cs156b/um/validation.dta', 'w')
		for line in f:
			if curr_num < num_lines * 0.9:
				b.write(line)
			else:
				c.write(line)
		b.close()
		c.close()

def get_training(dir):
	'''Takes filename of data file

	   Returns: input_data, output_data, full_data
	'''

	num_lines = sum(1 for line in open(dir))
	data = np.zeros((num_lines, 4))
	with open(dir) as f:
		cur_line = 0
		for line in f:
			line_split = line.split(' ')
			for i in range(4):
				data[cur_line][i] = int(line_split[i])

			cur_line += 1
			# TODO: for debugging
			if cur_line % 1000000 == 0:
				print (float(cur_line) / num_lines) * 100, '%% loaded'

	return data[:, :3][:].astype(int), data[:, 3:][:].astype(int), data.astype(int)

def get_validation_monitor(features, labels):
	'''Returns validation monitor object for training monitor'''
	validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall)
	}
	return tf.contrib.learn.monitors.ValidationMonitor(features,
													   labels,
													   every_n_steps=1)

def create_model_dir(base_dir, params):
	'''Returns model directory path'''
	for key in params:
		base_dir += '_' + key + str(params[key])
	return base_dir

def run_hyperparam_search(func, dimensions, n_calls, params):
	'''Calls skopt gp_minimize to do bayesian hyperparameter search. Returns search result'''
	return gp_minimize(func=func,
					   dimensions=dimensions,
					   acq_func='EI',
					   n_calls=n_calls,
					   x0=params)
