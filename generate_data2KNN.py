import utils
import numpy as np


um_data_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta"
um_idx_filename = "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.idx"


full_training1 = utils.get_from_alldta(um_data_filename, um_idx_filename, [1,2,3,4])



training1_filename =    '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/full_training.txt'


utils.write_np_arr_to_file(full_training1, training1_filename)


# 99666408 rows total
