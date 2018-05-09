'''Code for utilities'''
import tensorflow as tf
import numpy as np
from skopt import gp_minimize
from itertools import izip

#### Tensorflow Utils ####
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

#### Data utils ####

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

	return data[:, :3][:].astype('float32'), data[:, 3:][:].astype('float32'), data.astype(int)

def get_from_alldta(alldta_filename, allidx_filename, indeces_to_get):
	'''
	Gets specified data lines from all.dta

	alldta_filename: filename (including path) of the all.dta file
	allidx_filename: filename (including path) of the all.idx file
	indeces_to_get: list or set containing the indeces desired. From the readme:
				1: base (96% of the training set picked at random, use freely)
				2: valid (2% of the training set picked at random, use freely)
				3: hidden (2% of the training set picked at random, use freely)
				4: probe (1/3 of the test set picked at random, use freely but carefully)
				5: qual (2/3 of the test set picked at random, for testing the results)


	returns: dimension (number of lines, 4) numpy array. lines have form: (user, movie, date, rating)
	'''
	num_lines_alldta = 102416306 # number of lines in all.dta
	num_lines_output = 0 # number of lines in the outputted dataset
	data = np.zeros((num_lines_alldta, 4))

	with open(alldta_filename) as f, open(allidx_filename) as indeces_file:
		cur_line = 0
		for line, idx_line in izip(f, indeces_file):
			line_split = line.split(' ')
			idx = int(idx_line)
			if idx in indeces_to_get:
				for i in range(4):
					data[num_lines_output][i] = int(line_split[i])
				num_lines_output += 1
			cur_line += 1

			# TODO: for debugging
			if cur_line % 1000000 == 0:
				print (float(cur_line) / num_lines_alldta) * 100, '%% loaded'

	# print 'loaded lines: ' + str(num_lines_output)
	return (data.astype(int))[:num_lines_output]

def write_np_arr_to_file(np_arr, filename):
	'''
	Writes 2D np array to a text file. Space seperating elements in row, newline seperating rows. Eg:

		row1col1 row1col2 ...
		row2col1 row2col2 ...
		.
		.
		.

	** Make sure the type of the np array is the type u want

	'''
	assert np_arr.ndim == 2

	print np_arr.shape

	with open(filename, 'w') as f:
		for line in np_arr:
			f.write((' '.join(str(e) for e in line)) + '\n')


def write_results_to_file(np_arr, filename):
	'''
	Writes the contents of a dimension (2749898) np array to file.
	File format is that accepted by the 156 server: Ratings with 3 decimal places, seperated by newlines
	'''
	assert np_arr.ndim == 1

	with open(filename, 'w') as f:
		for line in np_arr:
			f.write(('%.3f' % line) + '\n')
