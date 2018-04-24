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
    common_viewers = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1)).astype(int) # cv[m][n], m < n
    m_avg = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))
    n_avg = np.zeros((NUM_MOVIES+1, NUM_MOVIES+1))

    start_user_idx = 0
    end_user_idx = 0
    while True:
        print float(start_user_idx) / len(UM_training_data), 'complete', start_user_idx, len(UM_training_data)
        while end_user_idx != len(UM_training_data) - 1 and UM_training_data[end_user_idx][0] == UM_training_data[end_user_idx + 1][0]:
            end_user_idx += 1

        # indexed [m][n]
        for i in range(start_user_idx, end_user_idx + 1):
            for j in range(i + 1, end_user_idx + 1):
                movie_idx_m = min(UM_training_data[i][1], UM_training_data[j][1]) # smaller movie number
                movie_idx_n = max(UM_training_data[i][1], UM_training_data[j][1]) # bigger movie number

                data_idx_m = i if UM_training_data[i][1] == movie_idx_m else j # row number with smaller movie number
                data_idx_n = i if UM_training_data[i][1] == movie_idx_n else j

                common_viewers[movie_idx_m][movie_idx_n] += 1
                m_avg[movie_idx_m][movie_idx_n] += UM_training_data[data_idx_m][3]
                n_avg[movie_idx_m][movie_idx_n] += UM_training_data[data_idx_n][3]

        if end_user_idx == len(UM_training_data) - 1:
            break
        else:
            end_user_idx += 1
            start_user_idx = end_user_idx

    print 'do div'
    for i in range(NUM_MOVIES + 1):
        if i % 500 == 0:
            print i
        for j in range(i, NUM_MOVIES + 1):
            # m_avg[i][j] = m_avg[i][j] / common_viewers[i][j] if common_viewers[i][j] != 0 else 0
            # n_avg[i][j] = n_avg[i][j] / common_viewers[i][j] if common_viewers[i][j] != 0 else 0
            if common_viewers[i][j] != 0:
                m_avg[i][j] = m_avg[i][j] / common_viewers[i][j]
                n_avg[i][j] = n_avg[i][j] / common_viewers[i][j]

    # f = open('common_viewser_test', 'w')
    # np.save(f, common_viewers)
    # f.close()
    print common_viewers[0:6, 0:6]
    print m_avg[0:6, 0:6]
    print n_avg[0:6, 0:6]




# TODO should move this somewhere else
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




training = utils.get_training(TRAIN_DATA_MINI, num_lines=NUM_ROWS_MINI)[2]
# UM_training = utils.get_training(UM_data, num_lines=NUM_ROWS_FULL)[2]

um_logic = utils.get_training(TRAIN_DATA_LOGIC_TEST, num_lines=5)[2]


to_predict = (1, 185, 2160) # correct answer is 1
params = {'minCV':16}

print('start train')

start_time = time.time()

train_knn(training)

end_time = time.time()

print 'done !!!!'
print str((start_time-end_time)/60.) + 'minutes'
# print(predict_knn(to_predict, training, params))
