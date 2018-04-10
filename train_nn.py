'''Code for training models'''
import numpy as np
import tensorflow as tf

import models
import utils

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta'
VALIDATION_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'
EPOCHS = 10
BATCH_SIZE = 64

# TODO: Fix config init
# CONFIG = tf.estimator.RunConfig(
#     		save_checkpoints_steps = 1000,  # Save checkpoints
#     		keep_checkpoint_max = 25,       # Retain the 100 most recent checkpoints.
# )
CONFIG = tf.estimator.RunConfig()

def train_input_fn(dir):
	'''Returns training input function'''
	features, labels = utils.get_training(TRAIN_DATA_MINI)
	return features, labels

def get_feature_column(features):
	'''Returns infered feature column from features'''
	feature_column = tf.contrib.learn.infer_real_valued_columns_from_input(features)
	return feature_column

def get_estimator(params, feature_column, CONFIG):
	'''Returns estimator object'''
	return models.DNN_classifier(params, feature_column, CONFIG)

def train():
	'''Start training'''
	params = {'hidden_units': '256',
			  'model_dir': './models/test',
			  'n_classes': 6,
			  'optimizer': 'adam',
			  'learning_rate': 0.001,
			  'activation_fn': 'softmax',
			  'dropout': 0.5}
	x, y = train_input_fn(TRAIN_DATA_MINI)
	feature_column = get_feature_column(x)
	estimator = get_estimator(params, feature_column, CONFIG)
	val_x, val_y = train_input_fn(VALIDATION_DATA_MINI)
	validation_monitor = utils.get_validation_monitor(val_x, val_y)
	estimator.fit(x = x,
				  y = y,
				  steps = EPOCHS,
				  batch_size = BATCH_SIZE,
				  monitors = [validation_monitor])

if __name__ == "__main__":
    train()