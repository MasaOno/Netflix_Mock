import numpy as np
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate

import utils

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta'
VALIDATION_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'

def get_training(train_dir, test_dir):
	'''Returns (train, test) data object'''
	reader = Reader(line_format='user item timestamp rating', sep=' ', rating_scale=(1, 5))
	train_data = Dataset.load_from_file(train_dir, reader=reader)
	test_data = Dataset.load_from_file(test_dir, reader=reader)
	return train_data.build_full_trainset(), test_data.build_full_trainset().build_testset()

def get_accuracy_temp(predictions):
	'''Returns accuracy from surprise model predictions'''
	err = 0
	for prediction in predictions:
		if prediction[2] == round(prediction[3]):
			err += 1
	return err / float(len(predictions))

def run_svd(data, params):
	'''Returns trained SVD model based on matrix factorization'''
	alg = SVD(biased=utils.get_param(params, 'biased'),
			  n_factors=utils.get_param(params, 'n_factors'),
			  n_epochs=utils.get_param(params, 'n_epochs'),
			  lr_all=utils.get_param(params, 'learning_rate'),
			  reg_all=utils.get_param(params, 'reg'))
	alg.fit(data)
	return alg

# TODO: save surprise model
def train():
	params = {'biased': True,
			  'n_factors': 40,
			  'n_epochs': 1500,
			  'learning_rate': 0.001,
			  'reg': 0.1}
	print 'Training SVD model...'
	print params
	print 'Loading data...'
	train_data, test_data = get_training(TRAIN_DATA_MINI, VALIDATION_DATA_MINI)
	print 'Factorizing...'
	model = run_svd(train_data, params)
	predictions = model.test(test_data)
	print predictions
	print get_accuracy_temp(predictions)

if __name__ == '__main__':
	train()