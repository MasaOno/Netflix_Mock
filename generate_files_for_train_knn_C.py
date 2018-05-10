# need to run this before running knn_no_python.c
#  have to change all the filenames....

import utils
import numpy as np

# TRAIN_DATA_LOGIC_TEST = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test.txt'
# TRAIN_DATA_MINI = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_test.txt'

# UM_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta' # sorted by user, movie
#
#
# print 1
# um_data_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta"
# um_idx_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.idx"
#
# # TODO, it would be faster to do ONE pass thru the data
# training1 = utils.get_from_alldta(um_data_filename, um_idx_filename, [1])
# valid2 = utils.get_from_alldta(um_data_filename, um_idx_filename, [2])
# hidden3 = utils.get_from_alldta(um_data_filename, um_idx_filename, [3])
# probe4 = utils.get_from_alldta(um_data_filename, um_idx_filename, [4])
# qual5 = utils.get_from_alldta(um_data_filename, um_idx_filename, [5])
#
# training1_filename =    '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt'
# valid2_filename =       '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt'
# hidden3_filename =      '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/hidden3.txt'
# probe4_filename =       '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/probe4.txt'
# qua5l_filename =        '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/qual5.txt'
#
# utils.write_np_arr_to_file(training1, training1_filename)
# utils.write_np_arr_to_file(valid2, valid2_filename)
# utils.write_np_arr_to_file(hidden3, hidden3_filename)
# utils.write_np_arr_to_file(probe4, probe4_filename)
# utils.write_np_arr_to_file(qual5, qua5l_filename)
#
# '''
# (94362233, 4)
# (1965045, 4)
# (1964391, 4)
# (1374739, 4)
#
# Traceback (most recent call last):
#   File "generate_files_for_train_knn_C.py", line 34, in <module>
#     utils.write_np_arr_to_file(probe4, probe4_filename)
# NameError: name 'qua5l' is not defined
# '''
# print 7


MU_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu/all.dta' # sorted by movie, user


print 1
mu_data_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu/all.dta"
mu_idx_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu/all.idx"

# TODO, it would be faster to do ONE pass thru the data
training1 = utils.get_from_alldta(mu_data_filename, mu_idx_filename, [1])
valid2 = utils.get_from_alldta(mu_data_filename, mu_idx_filename, [2])
hidden3 = utils.get_from_alldta(mu_data_filename, mu_idx_filename, [3])
probe4 = utils.get_from_alldta(mu_data_filename, mu_idx_filename, [4])
qual5 = utils.get_from_alldta(mu_data_filename, mu_idx_filename, [5])

print 2


training1_filename =    '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu_other/training1.txt'
valid2_filename =       '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu_other/valid2.txt'
hidden3_filename =      '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu_other/hidden3.txt'
probe4_filename =       '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu_other/probe4.txt'
qua5l_filename =        '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu_other/qual5.txt'

utils.write_np_arr_to_file(training1, training1_filename)
utils.write_np_arr_to_file(valid2, valid2_filename)
utils.write_np_arr_to_file(hidden3, hidden3_filename)
utils.write_np_arr_to_file(probe4, probe4_filename)
utils.write_np_arr_to_file(qual5, qua5l_filename)

'''
(94362233, 4)
(1965045, 4)
(1964391, 4)
(1374739, 4)
(2749898, 4)


'''
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
