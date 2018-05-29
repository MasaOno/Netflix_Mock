'''
Movie-to-movie Pearson's r-based KNN
From: http://dmnewbie.blogspot.com/2007/09/greater-collaborative-filtering.html
'''

import numpy as np

import utils
import time

# TODO can move these constants
NUM_ROWS_FULL = 102416306
NUM_ROWS_MINI = 1000000

NUM_USERS = 458293 # 1 indexed
NUM_MOVIES = 17770 # 1 indexed

# TRAIN_DATA_MINI = '/Users/masaono/Desktop/cs156b/um/train_mini.dta' # Masa
TRAIN_DATA_LOGIC_TEST = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test.txt'
TRAIN_DATA_MINI = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_test.txt'

MU_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/mu/all.dta' # sorted by movie, user
UM_data = '/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta' # sorted by user, movie


def train_knn(UM_training_data):
    # Pre-calculated statistics: train once and predict forever
    # For a Movie pair (m, n) with common viewers
    # arr[m][n], m < n
    common_viewers = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1)).astype(int)
    m_sum = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))
    n_sum = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))
    mn_sum = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))
    mm_sum = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))
    nn_sum = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))

    # TODO calculate mAvg for all viewers (not common viewer). can do it in first for loop.

    start_user_idx = 0
    end_user_idx = 0
    while True:
        # TODO: delete for production
        print float(start_user_idx) / len(UM_training_data), 'complete', start_user_idx, len(UM_training_data)

        # Move the end_user_idx pointer to the final rating made by that user
        while end_user_idx != len(UM_training_data) - 1 and UM_training_data[end_user_idx][0] == UM_training_data[end_user_idx + 1][0]:
            end_user_idx += 1

        # Iterate through the movies rated by the current user
        # indexed [m][n]
        for i in range(start_user_idx, end_user_idx + 1):
            for j in range(i + 1, end_user_idx + 1):
                # We need this bc we fill only half the matrix to avoid duplicated data

                # TODO: 5/1/ opt 1. take advantage of data ordering
                # movie_idx_m = min(UM_training_data[i][1], UM_training_data[j][1]) # smaller movie number
                # movie_idx_n = max(UM_training_data[i][1], UM_training_data[j][1]) # bigger movie number
                movie_idx_m = UM_training_data[i][1] # smaller movie number
                movie_idx_n = UM_training_data[j][1]

                # TODO: 5/1/ opt 1. take advantage of data ordering
                # row number of movie
                # data_idx_m = i if UM_training_data[i][1] == movie_idx_m else j
                # data_idx_n = i if UM_training_data[i][1] == movie_idx_n else j
                ####
                # data_idx_m = i
                # data_idx_n = j


                common_viewers[movie_idx_m][movie_idx_n] += 1
                m_sum[movie_idx_m][movie_idx_n] += UM_training_data[i][3]
                n_sum[movie_idx_m][movie_idx_n] += UM_training_data[j][3]
                mn_sum[movie_idx_m][movie_idx_n] += UM_training_data[i][3] * UM_training_data[j][3]
                mm_sum[movie_idx_m][movie_idx_n] += UM_training_data[i][3] * UM_training_data[i][3]
                nn_sum[movie_idx_m][movie_idx_n] += UM_training_data[j][3] * UM_training_data[j][3]

        if end_user_idx == len(UM_training_data) - 1:
            break
        else:
            end_user_idx += 1
            start_user_idx = end_user_idx

    # print 'do div'
    # for i in range(NUM_MOVIES + 1):
    #     if i % 500 == 0:
    #         print i
    #     for j in range(i, NUM_MOVIES + 1):
    #         # m_avg[i][j] = m_avg[i][j] / common_viewers[i][j] if common_viewers[i][j] != 0 else 0
    #         # n_avg[i][j] = n_avg[i][j] / common_viewers[i][j] if common_viewers[i][j] != 0 else 0
    #         if common_viewers[i][j] != 0:
    #             m_avg[i][j] = m_avg[i][j] / common_viewers[i][j]
    #             n_avg[i][j] = n_avg[i][j] / common_viewers[i][j]

    # f = open('common_viewser_test', 'w')
    # np.save(f, common_viewers)
    # f.close()
    print common_viewers[0:6, 0:6]
    print m_sum[0:6, 0:6]
    print n_sum[0:6, 0:6]
    print mn_sum[0:6, 0:6]
    print mm_sum[0:6, 0:6]
    print nn_sum[0:6, 0:6]




def predict_knn(to_predict, training_data, params):
    '''
    TODO: doesnt do shit rn


    Predicts movie rating for a given movie, user, and date. (date is irrelevant)

    to_predict: tuple of form (user, movie, date)
    UM_training_data: (num_rows, 4) shaped np array. Each row has form (user, movie, date, rating)
    params: Dict with the following params
                minCV: minimum number of common viewers required.

    '''

    L0 = set() # All movies rated by the viewer
    for training_row in training_data:
        if training_row[0] == to_predict[0]: # Movie rated by to_predict user
            # Count viewers that rated both to_predict movie and


            L0.add(training_row[1])


    movieNeighbors = []



# print('Training on 1mil row UM dataset')
# training = utils.get_training(TRAIN_DATA_MINI)[2]

# print('Training on 100mil row UM dataset')
# training = utils.get_training(UM_data, num_lines=NUM_ROWS_FULL)[2]

print('Training on 5 row logic test dataset')
training = utils.get_training(TRAIN_DATA_LOGIC_TEST)[2]


to_predict = (1, 185, 2160) # correct answer is 1
params = {'minCV':16}

print('start train')

start_time = time.time()

train_knn(training)

end_time = time.time()

print 'done !!!!'
print str((end_time-start_time)/60.) + ' minutes'
# print(predict_knn(to_predict, training, params))
