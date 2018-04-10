import numpy as np
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate

import utils

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta'

def get_training(dir):
	'''Returns (train, test) data object'''
	reader = Reader(line_format='user item timestamp rating', sep=' ', rating_scale=(1, 5))
	train_data = Dataset.load_from_file(dir, reader=reader)
	return train_data.build_full_trainset()

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
			  'n_factors': 20,
			  'n_epochs': 10,
			  'learning_rate': 0.001,
			  'reg': 0.1}
	print 'Training SVD model...'
	print params
	print 'Loading data...'
	train_data = get_training(TRAIN_DATA_MINI)
	print 'Factorizing...'
	model = run_svd(train_data, params)

if __name__ == '__main__':
	train()