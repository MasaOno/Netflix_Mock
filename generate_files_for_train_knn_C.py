# need to run this before running knn_no_python.c
#  have to change all the filenames....

import utils
import numpy as np

TRAIN_DATA_LOGIC_TEST = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test.txt'
TRAIN_DATA_MINI = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_test.txt'

# MU_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu/all.dta' # sorted by movie, user
UM_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta' # sorted by user, movie


print 1
um_data_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta"
um_idx_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.idx"

training1 = utils.get_from_alldta(um_data_filename, um_idx_filename, [1])
print 2
valid2 = utils.get_from_alldta(um_data_filename, um_idx_filename, [2])
print 3
qua5l = utils.get_from_alldta(um_data_filename, um_idx_filename, [5])
print 4
training1_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt'
valid2_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt'
qua5l_filename = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/qual5.txt'

utils.write_np_arr_to_file(training1, training1_filename)
print 5
utils.write_np_arr_to_file(valid2, valid2_filename)
print 6
utils.write_np_arr_to_file(qua5l, qua5l_filename)
print 7


# #
# write_np_arr_to_file_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/2d_arr.txt"
# t = np.array([[1,2,3], [3,3,3], [6,6,6], [69,69,102]])
# utils.write_np_arr_to_file(t, write_np_arr_to_file_filename)
#
#
# write_results_to_filefilename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/1d_arr.txt"
# fofo = np.array([2.09929292922992,2,3,4,69,69])
# utils.write_results_to_file(fofo, write_results_to_filefilename)
