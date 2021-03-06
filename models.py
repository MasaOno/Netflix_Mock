'''Code for different models'''
import tensorflow as tf

import utils

def DNN_regressor(params, model_dir, feature_columns, config):
	'''Returns DNN estimator object'''
	hidden_units = params['layers'] * [params['units']]
	weight_column_name = utils.get_param(params, 'weight_column_name')
	optimizer = utils.get_optimizer(utils.get_param(params, 'optimizer'), params['learning_rate'])
	activation_fn = utils.get_activation(utils.get_param(params, 'activation_fn'))
	dropout = float(utils.get_param(params, 'dropout'))
	gradient_clip_norm = utils.get_param(params, 'gradient_clip_norm')
	enable_centered_bias = False # keep false
	feature_engineering_fn = utils.get_param(params, 'feature_engineering_fn')
	embedding_lr_multipliers = utils.get_param(params, 'embedding_lr_multipliers')
	input_layer_min_slice_size = utils.get_param(params, 'input_layer_min_slice_size')
	label_keys = utils.get_param(params, 'label_keys')

	return tf.contrib.learn.DNNRegressor(hidden_units = hidden_units,
										  feature_columns = feature_columns,
										  model_dir = model_dir,
										  weight_column_name = weight_column_name,
										  optimizer = optimizer,
										  activation_fn = activation_fn,
										  dropout = dropout,
										  gradient_clip_norm = gradient_clip_norm,
										  enable_centered_bias = enable_centered_bias,
										  config = config,
										  feature_engineering_fn = feature_engineering_fn,
										  embedding_lr_multipliers = embedding_lr_multipliers,
										  input_layer_min_slice_size = input_layer_min_slice_size)

def tensor_forest(params):
	'''Returns tensorforest estimator object'''
	num_classes = int(utils.get_param(params, 'num_classes'))
	num_features = int(utils.get_param(params, 'num_features'))
	regression = bool(utils.get_param(params, 'regression'))
	num_trees = int(utils.get_param(params, 'num_trees'))
	max_nodes = int(utils.get_param(params, 'max_nodes'))
	tensor_forest_params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(num_classes = num_classes,
																					   num_features = num_features,
																					   regression = regression,
																					   num_trees = num_trees,
																					   max_nodes = max_nodes)
	return tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(tensor_forest_params)