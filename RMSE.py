import math
from itertools import izip



def validation_rmse(dataset_filename, prediction_filename):
	'''
	dataset_filename: file is of form:
			user movie date rating
			user movie date rating
			user movie date rating
			...

	prediction_filename: file is of form:
			ratings
			ratings
			ratings
			...
	'''
	with open(dataset_filename) as dataset, open(prediction_filename) as preds:
		cur_line = 0

		cur_sum = 0.

		for line, prediction_str in izip(dataset, preds):
			actual = float(line.split(' ')[3])
			prediction = float(prediction_str)

			if not math.isnan(prediction):
				cur_sum += (actual - prediction) ** 2
				cur_line += 1

			if math.isnan(prediction):
				print 'prediction_str',prediction_str
				print prediction, actual


			# if cur_line % 10000 == 0:
			# 	# print cur_line/1965045.
			# 	print actual, prediction

		print 'cur_sum', cur_sum
		print 'cur_line', cur_line


		return math.sqrt(cur_sum / float(cur_line))



# dataset_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt'
dataset_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/probe4.txt'

# KNN
# prediction_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_VALIDATION.txt' # Validation Error
# prediction_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_PROBE.txt"

# SVD
prediction_filename_prd = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/SVD_elo_UM_output_PROBE.txt'
prediction_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/NEW_SVD_elo_UM_output_PROBE.txt'
prediction_filename_old = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/OLD_SVD_elo_UM_output_PROBE.txt'




# prediction_filename ='/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_TRAINING_PREDICTION.txt' # Training Errrp

prod_rmse = validation_rmse(dataset_filename, prediction_filename_prd)
opt_rmse = validation_rmse(dataset_filename, prediction_filename)
opt_rmse_old = validation_rmse(dataset_filename, prediction_filename_old)

print 'opt rmse vs rmse old: ', opt_rmse , opt_rmse_old

print 'prod rmse: ', prod_rmse
