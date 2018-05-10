import numpy as np
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp
from surprise.model_selection import cross_validate

import utils

TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'
VALIDATION_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/validation_mini.dta'

# TRAIN_DATA_FULL = '/Users/masaono/Desktop/cs156b/um/train_all.dta'
# VALIDATION_DATA_FULL = '/Users/masaono/Desktop/cs156b/um/validation.dta'
# TEST_DATA_FULL = '/Users/masaono/Desktop/cs156b/um/qual.dta'

TRAIN_DATA_FULL = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt'
VALIDATION_DATA_FULL = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/probe4.txt'
TEST_DATA_FULL = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/qual.dta'

def get_training(train_dir, validation_dir, test_dir):
	'''Returns (train, test) data object'''
	reader = Reader(line_format='user item timestamp rating', sep=' ', rating_scale=(1, 5))
	reader_test = Reader(line_format='user item rating', sep=' ', rating_scale=(1, 2243))
	train_data = Dataset.load_from_file(train_dir, reader=reader)
	validation_data = Dataset.load_from_file(validation_dir, reader=reader)
	test_data = Dataset.load_from_file(test_dir, reader=reader_test)
	return train_data.build_full_trainset(), validation_data.build_full_trainset().build_testset(), test_data.build_full_trainset().build_testset()

# TODO: Place temp functions in evaluate.py
# =================================================================
def get_accuracy_temp(predictions):
	'''Returns accuracy from surprise model predictions'''
	err = 0
	for prediction in predictions:
		if prediction[2] == round(prediction[3]):
			err += 1
	return err / float(len(predictions))

def get_rmse_temp(predictions):
	'''Returns accuracy from surprise model predictions'''
	s = 0.0
	val_length = 0
	for prediction in predictions:
		s += ((prediction[2] - float(prediction[3])) ** 2)
		val_length += 1
	return float(s) /  float(val_length)

def print_predictions_temp(predictions, output_dir):
	'''Creates prediction ourput file from predictions'''
	with open(output_dir, 'w') as f:
		for prediction in predictions:
			f.write(str(round(prediction[3], 3)) + '\n')
# =================================================================

def run_svd(data, params, svdpp = False):
	'''Returns trained SVD model based on matrix factorization'''
	if svdpp:
		alg = SVDpp(n_factors=utils.get_param(params, 'n_factors'),
			  n_epochs=utils.get_param(params, 'n_epochs'),
			  lr_all=utils.get_param(params, 'learning_rate'),
			  reg_all=utils.get_param(params, 'reg'),
			  verbose=True)
	else:
		alg = SVD(biased=utils.get_param(params, 'biased'),
				  n_factors=utils.get_param(params, 'n_factors'),
				  n_epochs=utils.get_param(params, 'n_epochs'),
				  lr_all=utils.get_param(params, 'learning_rate'),
				  reg_all=utils.get_param(params, 'reg'),
				  verbose=True)
	alg.fit(data)
	return alg

# TODO: save surprise model
def train():
	params = {'biased': True,
			  'n_factors': 40,
			  'n_epochs': 50,
			  'learning_rate': 0.001,
			  'reg': 0.1}
	model_dir = './models/svd/'
	print 'Training SVD model...'
	print params
	print 'Loading data...'
	train_data, validation_data, test_data = get_training(TRAIN_DATA_FULL, VALIDATION_DATA_FULL, TEST_DATA_FULL)
	print 'Factorizing...'
	model = run_svd(train_data, params, svdpp = True)
	# Get rmse from validation
	predictions = model.test(validation_data)
	print 'rmse: ' + str(get_rmse_temp(predictions))
	# Print predictions
	predictions = model.test(test_data)
	print_predictions_temp(predictions, './SVD_data/svdpp.txt')
	# Save model
	surprise.dump.dump(utils.create_model_dir(model_dir, params), algo = model)

if __name__ == '__main__':
	train()
