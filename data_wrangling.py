
import numpy as np
import sys

filename = 'data/um/all.dta'
n_lines = 102416306

def load_data(filename, num_lines):
    '''
    Loads data into a (num_lines, 4) dimension np array

    The 4 colums are: user number, movie number, date number, rating
    '''
    data = np.zeros((num_lines, 4))

    f = open(filename)

    cur_line = 0
    for l in f:
        txt_line = l.split(' ')
        data[cur_line][0] = int(txt_line[0])
        data[cur_line][1] = int(txt_line[1])
        data[cur_line][2] = int(txt_line[2])
        data[cur_line][3] = int(txt_line[3])

        cur_line += 1

        if cur_line % 10000 == 0:
            bar_num = int((float(cur_line)/float(num_lines)) * 10)
            bar =('#' * bar_num) + (' ' * (10 - bar_num)) + '|'
            print '\r', (float(cur_line)/float(num_lines))*100, '%% loaded ', '\t', bar,
            sys.stdout.flush()

    print 'DONE'
    return data

def extract_1mil_um():
    '''
    Extracts first million TRAINING rows (not test) from the um dataset
    '''
    filename = 'data/um/all.dta'
    lines = 0
    abridged_file = open('data/1mil_testes', 'w')
    f = open(filename)
    for l in f:
        txt_line = l.split(' ')
        if txt_line[3] != '0\n': # only training data
            abridged_file.write(l)
            lines += 1

            if lines == 1000000:
                break

    print 'done'

extract_1mil_um()
ddd = load_data('data/1mil_testes', 1000000)

print ddd[0]
