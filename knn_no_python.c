// C code for KNN

// TODO: for presentation: python expeced runtime 10hr

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO choose which data directory u wanna use
// ETHAN_UM: 1
// MASA_UM: 2
// ETHAN_UM_ONE_MIL: 3
// ETHAN_UM_LOGIC: 4
#define TRAINING_DATA_VERSION 3

// All data is in UM form
#if TRAINING_DATA_VERSION == 1 // Full UM dataset on Ethan's machine
  #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/all.dta"
  #define TRAINING_NUM_ROWS 94362233
#elif TRAINING_DATA_VERSION == 2 // Full UM dataset on Masa's machine
  #define TRAINING_DATA_PATH "" // TODO: masa needs to fill out
  #define TRAINING_NUM_ROWS 94362233
#elif TRAINING_DATA_VERSION == 3 // 1 million row test UM dataset
  #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_test.txt"
  #define TRAINING_NUM_ROWS 1000000
#elif TRAINING_DATA_VERSION == 4 // 5 row test dataset
  #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test.txt"
  #define TRAINING_NUM_ROWS 5
#endif

#define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/qual.dta"
#define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_output.txt"
#define TEST_NUM_ROWS 2749898

// #define VALID_NUM_ROWS 1965045

#define NUM_USERS 458293 // 1 indexed
#define NUM_MOVIES 17770 // 1 indexed
#define ELEMS_IN_ROW 4 // 4 elements in a row: user, movie, date, rating
#define ELEMS_IN_TEST_ROW 3 // 4 elements in a row: user, movie, date


////// KNN PARAMS
#define MIN_CV 16
#define MAX_NEIGHBORS 50

// For each movie pair (m, n) where m < n
// Statistics on users who rated both m and n
struct PearsonIntermediate {
  unsigned int common_viewers;
  float m_sum;
  float n_sum;
  float mn_sum;
  float mm_sum;
  float nn_sum;
};


int main() {
   // training_data[row rum][col num]
   // cols are (user, movie, date, rating)
   int * training_data = (int*)malloc(TRAINING_NUM_ROWS * ELEMS_IN_ROW * sizeof(int)); // TODO free
   printf("allocated %f GB array training_data\n", (TRAINING_NUM_ROWS * ELEMS_IN_ROW * sizeof(int))/1000000000.);

  // Pre-calculated statistics for a Movie pair (m, n) with common viewers
  // pearson_intermediates[m * ELEMS_IN_ROW + n], m < n
  struct PearsonIntermediate * pearson_intermediates = (struct PearsonIntermediate *)malloc((NUM_MOVIES + 1) * (NUM_MOVIES + 1) * sizeof(struct PearsonIntermediate)); // TODO free
  int l = 0;
  for (l = 0; l < (NUM_MOVIES + 1) * (NUM_MOVIES + 1); l++) {
    pearson_intermediates[l].common_viewers = 0;
    pearson_intermediates[l].m_sum = 0.;
    pearson_intermediates[l].n_sum = 0.;
    pearson_intermediates[l].mn_sum = 0.;
    pearson_intermediates[l].mm_sum = 0.;
    pearson_intermediates[l].nn_sum = 0.;
  }
  printf("allocated and zeroed %f GB pearson_intermediates array\n", ((NUM_MOVIES + 1) * (NUM_MOVIES + 1) * sizeof(struct PearsonIntermediate))/1000000000.);

  // Sum of each movie's ratings from all users (not common viewers from another movie)
  int * m_sums = (int*)malloc((NUM_MOVIES + 1) * sizeof(int)); // TODO free
  memset(m_sums, 0, (NUM_MOVIES + 1) * sizeof(int));
  printf("allocated and zeroed %f GB m_sums array\n", ((NUM_MOVIES + 1) * sizeof(int))/1000000000.);

  // For each user, the index where their data starts
  int * user_start_idxs = (int*)malloc((NUM_USERS + 1) * sizeof(int)); // TODO free
  memset(user_start_idxs, 0, (NUM_USERS + 1) * sizeof(int));
  printf("allocated and zeroed %f GB user_start_idxs array\n", ((NUM_USERS + 1) * sizeof(int))/1000000000.);

  //// Read in training data
  FILE * training_data_file;
  char line_buf [100]; // holds each line
  training_data_file = fopen(TRAINING_DATA_PATH, "r");
  if (!training_data_file) {
    return 1;
  }
  // For each line write the 4 columns
  char num_buf [20];
  int line_buf_idx = 0;
  int num_buf_idx = 0;
  int i = 0;
  int line_num = 0;
  while (fgets(line_buf, 100, training_data_file) != NULL) { // for each line
    if (line_num % 100000 == 0) {
      printf("\r%f percent data loaded", ((float)(line_num)/(float)(TRAINING_NUM_ROWS))*100);
    }

    // each line is: user<space>movie<space>date<space>rating<newline>
    for (i = 0; i < 4; i++) {
      while (line_buf[line_buf_idx] != ' ' && line_buf[line_buf_idx] != '\n') {
        num_buf[num_buf_idx] = line_buf[line_buf_idx];
        line_buf_idx += 1;
        num_buf_idx += 1;
      }
      num_buf[num_buf_idx] = ' ';
      training_data[line_num * ELEMS_IN_ROW + i] = atoi(num_buf);
      line_buf_idx += 1;
      num_buf_idx = 0;
    }
    line_buf_idx = 0;
    line_num += 1;
  }
  fclose(training_data_file);
  printf("\n===============\n");

  //// compute pearson intermediates
  int start_user_idx = 0;
  int end_user_idx = 0;

  while (1) {
    printf("\r%f pearson calc complete", (float)start_user_idx / (float)TRAINING_NUM_ROWS); // TODO delete for prod

    // Move the end_user_idx pointer to the final rating made by that user
    while (end_user_idx != TRAINING_NUM_ROWS - 1 && training_data[end_user_idx * ELEMS_IN_ROW] == training_data[(end_user_idx + 1) * ELEMS_IN_ROW]) {
      end_user_idx += 1;
    }

    user_start_idxs[training_data[start_user_idx * ELEMS_IN_ROW + 0]] = start_user_idx;

    // Iterate through the movies rated by the current user
    // indexed [m][n]
    int j = 0;
    for (i = start_user_idx; i < end_user_idx + 1; i++) {
      int movie_idx_m = training_data[i * ELEMS_IN_ROW + 1];
      m_sums[movie_idx_m] = training_data[i * ELEMS_IN_ROW + 3];
      for (j = i + 1; j < end_user_idx + 1; j++) {
        int movie_idx_n = training_data[j * ELEMS_IN_ROW + 1];

        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].common_viewers += 1;
        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].m_sum += training_data[i * ELEMS_IN_ROW + 3];
        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].n_sum += training_data[j * ELEMS_IN_ROW + 3];
        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].mn_sum += training_data[i * ELEMS_IN_ROW + 3] * training_data[j * ELEMS_IN_ROW + 3];
        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].mm_sum += training_data[i * ELEMS_IN_ROW + 3] * training_data[i * ELEMS_IN_ROW + 3];
        pearson_intermediates[movie_idx_m * ELEMS_IN_ROW + movie_idx_n].nn_sum += training_data[j * ELEMS_IN_ROW + 3] * training_data[j * ELEMS_IN_ROW + 3];
      }
    }

    if (end_user_idx == TRAINING_NUM_ROWS - 1) {
      break;
    }
    else {
      end_user_idx += 1;
      start_user_idx = end_user_idx;
    }
  }

  //// Compute pearson coef (r) -> rLower -> weight TODO

  //// End of "training" part ////
  float * test_data_output = (float*)malloc(TEST_NUM_ROWS * sizeof(float)); // TODO free

  // Load test set input
  int * test_data = (int*)malloc(TEST_NUM_ROWS * ELEMS_IN_TEST_ROW * sizeof(int)); // TODO free
  printf("allocated %f GB array test_data\n", (TEST_NUM_ROWS * ELEMS_IN_TEST_ROW * sizeof(int))/1000000000.);

  FILE * test_data_file;
  test_data_file = fopen(TEST_INPUT_PATH, "r");
  if (!test_data_file) {
    return 1;
  }
  // For each line write the 4 columns
  line_buf_idx = 0;
  num_buf_idx = 0;
  i = 0;
  line_num = 0;
  while (fgets(line_buf, 100, test_data_file) != NULL) { // for each line
    if (line_num % 100000 == 0) {
      printf("\r%f percent test data loaded", ((float)(line_num)/(float)(TEST_NUM_ROWS))*100);
    }

    // each line is: user<space>movie<space>date<newline>
    for (i = 0; i < ELEMS_IN_TEST_ROW; i++) {
      while (line_buf[line_buf_idx] != ' ' && line_buf[line_buf_idx] != '\n') {
        num_buf[num_buf_idx] = line_buf[line_buf_idx];
        line_buf_idx += 1;
        num_buf_idx += 1;
      }
      num_buf[num_buf_idx] = ' ';
      training_data[line_num * ELEMS_IN_TEST_ROW + i] = atoi(num_buf);
      line_buf_idx += 1;
      num_buf_idx = 0;
    }
    line_buf_idx = 0;
    line_num += 1;
  }
  fclose(test_data_file);
  printf("\n===============\n");

  // Iterate throuhg all test inputs
  for (i = 0; i < TEST_NUM_ROWS; i++) {
    // mAvg, nAvg, rRaw, weight, ratingN
  }







}
