// C code for KNN

// TODO: for presentation: python expeced runtime 10hr
/*
        FUcked ass bug 1: flipped order for Qsort for weights

*/



// submit code: pdnijpwr

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// TODO choose which data directory u wanna use
// ETHAN_UM: 1
// MASA_UM: 2
// ETHAN_UM_ONE_MIL: 3
// ETHAN_UM_LOGIC: 4
// ETHAN_UM_LOGIC2: 5
#define TRAINING_DATA_VERSION 1

// All data is in UM form
#if TRAINING_DATA_VERSION == 1 // Full UM dataset on Ethan's machine
  #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt"
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
  #elif TRAINING_DATA_VERSION == 5 // 20 row test dataset
    #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test.txt"
    #define TRAINING_NUM_ROWS 20
#endif

// // For test submission
// #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/qual.dta"
// #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_subToServer_output.txt"
// #define TEST_NUM_ROWS 2749898

// For validation set: index 2
#define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt"
#define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_VALIDATION.txt"
#define TEST_NUM_ROWS 1965045

// // For hidden set, idx3

// // For probe set, idx4

// // For training error, RMSE
// #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt"
// #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_TRAINING_PREDICTION.txt"
// #define TEST_NUM_ROWS 94362233


#define NUM_USERS 458293 // 1 indexed
#define NUM_MOVIES 17770 // 1 indexed
#define ELEMS_IN_ROW 4 // 4 elements in a row: user, movie, date, rating
#define ELEMS_IN_TEST_ROW 3 // 4 elements in a row: user, movie, date


////// KNN PARAMS
#define MIN_CV 16
#define MAX_NEIGHBORS 400

#define COMPUTE_OR_LOAD 1 // 1 -> compute, 0 -> load


// For each movie pair (m, n) where m < n
// Statistics on users who rated both m and n
struct PearsonIntermediate {
  unsigned int common_viewers;
  float m_sum;
  float n_sum;
  float mn_sum;
  float mm_sum;
  float nn_sum;

  // Calcualted in a second pass
  float r_raw;
  float weight;
};

struct MovieNeighbor {
   float m_avg;
   float nAvg;
   float rRaw;
   float weight;
   float rating_n;
};

int min(int a, int b) {
  if (a < b) {
    return a;
  }
  else {
    return b;
  }
}

int max(int a, int b) {
  if (a < b) {
    return b;
  }
  else {
    return a;
  }
}

float round_pos_float(float num) {
  return (float)((int)(num + .5));
}

/*
For comparing 2 Movie neighbors

Returns:
  lt 0  : if a goes before b, a.weight > b.weight
  0     : if same
  gt 0  : if b goes before a, a.weight < b.weight
*/
int weight_compare(const void * a, const void * b) {
  return -(((struct MovieNeighbor *)a)->weight - ((struct MovieNeighbor *)b)->weight);
}

void load_training(int * training_data) {
  //// Read in training data
  FILE * training_data_file;
  char line_buf [100]; // holds each line
  training_data_file = fopen(TRAINING_DATA_PATH, "r");
  if (!training_data_file) {
    return;
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
}


// TODO: move this constant
#define PEARSON_DIR "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/KNN_data/"
#define USER_STARTS_IDX_FILENAME "user_start_idxs.data"
#define M_SUMS_FILENAME "m_sums.data"
#define PEARSON_INTERMEDIATES_FILENAME "pearson_intermediates.data"

void write_pearson_intermediate(int * user_start_idxs, int * m_sums, struct PearsonIntermediate * pearson_intermediates) {
  printf("WRITING CALCULATED PEARSON INTERMEDIATES\n");

  FILE *f;

  f = fopen(PEARSON_DIR USER_STARTS_IDX_FILENAME, "wb");
  fwrite(user_start_idxs, sizeof(int), NUM_USERS + 1, f);
  fclose(f);

  f = fopen(PEARSON_DIR M_SUMS_FILENAME, "wb");
  fwrite(m_sums, sizeof(int), NUM_MOVIES + 1, f);
  fclose(f);

  f = fopen(PEARSON_DIR PEARSON_INTERMEDIATES_FILENAME, "wb");
  fwrite(pearson_intermediates, sizeof(struct PearsonIntermediate), ((NUM_MOVIES + 1)*(NUM_MOVIES + 1)), f);
  fclose(f);

}

void load_pearson_intermediate(int * user_start_idxs, int * m_sums, struct PearsonIntermediate * pearson_intermediates) {
  printf("LOADING CALCULATED PEARSON INTERMEDIATES\n");

  FILE *ifp;

  ifp = fopen(PEARSON_DIR USER_STARTS_IDX_FILENAME, "rb");
  fread(user_start_idxs, sizeof(int), NUM_USERS + 1, ifp);
  fclose(ifp);

  ifp = fopen(PEARSON_DIR M_SUMS_FILENAME, "rb");
  fread(m_sums, sizeof(int), NUM_MOVIES + 1, ifp);
  fclose(ifp);

  ifp = fopen(PEARSON_DIR PEARSON_INTERMEDIATES_FILENAME, "rb");
  fread(pearson_intermediates, sizeof(struct PearsonIntermediate), ((NUM_MOVIES + 1)*(NUM_MOVIES + 1)), ifp);
  fclose(ifp);
}

void compute_pearson_intermediate(const int * training_data, int * user_start_idxs, int * m_sums, struct PearsonIntermediate * pearson_intermediates) {
  int i, j;

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

  // Calculate pearson coef and weight
  for (i = 1; i < NUM_MOVIES + 1; i++) {
    for (j = i + 1; j < NUM_MOVIES + 1; j++){
      int idx = i * ELEMS_IN_ROW + j;
      pearson_intermediates[idx].r_raw = ((pearson_intermediates[idx].common_viewers * pearson_intermediates[idx].mn_sum) - (pearson_intermediates[idx].m_sum * pearson_intermediates[idx].n_sum)) / (sqrt(pearson_intermediates[idx].common_viewers * pearson_intermediates[idx].mm_sum - pearson_intermediates[idx].m_sum * pearson_intermediates[idx].m_sum) * sqrt(pearson_intermediates[idx].common_viewers * pearson_intermediates[idx].nn_sum - pearson_intermediates[idx].n_sum * pearson_intermediates[idx].n_sum));

      float r_lower;
      if (pearson_intermediates[idx].common_viewers <= 3) {
        r_lower = 0.;
      }
      else {
        // OLd formula
        // r_lower = 0.5 * log((1. + pearson_intermediates[idx].r_raw) / (1. - pearson_intermediates[idx].r_raw)) - (1.96 / sqrt(pearson_intermediates[idx].common_viewers - 3));

        // TODO try this new shit out
        r_lower = tanh(0.5 * log((1. + pearson_intermediates[idx].r_raw) / (1. - pearson_intermediates[idx].r_raw)) - (1.96 / sqrt(pearson_intermediates[idx].common_viewers - 3)));

      }


      // rLower may end up having a different sign than rRaw, in this case, I chose to set rLower to 0
      if ((r_lower < 0) != (pearson_intermediates[idx].r_raw < 0)) {
        r_lower = 0.;
      }

      pearson_intermediates[idx].weight = r_lower * r_lower * log(pearson_intermediates[idx].common_viewers);
    }
  }
}

void load_test(int * test_data_input) {
  FILE * test_data_file;
  char line_buf [100]; // holds each line
  test_data_file = fopen(TEST_INPUT_PATH, "r");
  if (!test_data_file) {
    return;
  }
  // For each line write the 4 columns
  int line_buf_idx = 0;
  int num_buf_idx = 0;
  char num_buf [20];
  int i = 0;
  int line_num = 0;
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

      // WTFFFFFF
      // training_data[line_num * ELEMS_IN_TEST_ROW + i] = atoi(num_buf); // TODO change was here
      test_data_input[line_num * ELEMS_IN_TEST_ROW + i] = atoi(num_buf); // TODO change was here
      //WTFFFFFF


      line_buf_idx += 1;
      num_buf_idx = 0;
    }
    line_buf_idx = 0;
    line_num += 1;
  }
  fclose(test_data_file);
  printf("\n===============\n");
}


void predict_test_set(const int * m_sums, const int * test_data_input, const int * user_start_idxs, const int * training_data, const struct PearsonIntermediate * pearson_intermediates, float * test_data_output) {
  int i, j;
  int start_user_idx;
  int end_user_idx;

  // Iterate through all test inputs
  for (i = 0; i < TEST_NUM_ROWS; i++) { // for each line test_data_input[i]
    if(i % 10000 == 0) { // TODO: for debugging
      printf("\r%f predections complete", ((float)i) / ((float)TEST_NUM_ROWS) );
    }

    int movie_m = test_data_input[i * ELEMS_IN_TEST_ROW + 1];
    int user_v = test_data_input[i * ELEMS_IN_TEST_ROW + 0];
    start_user_idx = user_start_idxs[user_v]; // where in training_data the ratings for user_v start
    end_user_idx = start_user_idx;
    while (end_user_idx != TRAINING_NUM_ROWS - 1 && training_data[end_user_idx * ELEMS_IN_ROW] == training_data[(end_user_idx + 1) * ELEMS_IN_ROW]) {
      end_user_idx += 1; // end_user_idx is the LAST idx of this user. NOT the idx of the start of the next user.
    }

    // TODO MUST FREE THIS!!! TODO TODO TODO
    struct MovieNeighbor * neighbors = (struct MovieNeighbor *)malloc((2 + end_user_idx - start_user_idx) * sizeof(struct MovieNeighbor));
    int num_neighbors = 0;
    for (j = start_user_idx; j < end_user_idx + 1; j++) { // for each movie rated by user_v
      int movie_n = training_data[j * ELEMS_IN_ROW + 1]; // neighboring movie
      if (pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].common_viewers >= MIN_CV) {
        neighbors[num_neighbors].m_avg = pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].m_sum / pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].common_viewers;
        neighbors[num_neighbors].nAvg = pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].n_sum / pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].common_viewers;
        neighbors[num_neighbors].rRaw = pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].r_raw;
        neighbors[num_neighbors].weight = pearson_intermediates[min(movie_m, movie_n) * ELEMS_IN_ROW + max(movie_m, movie_n)].weight;
        neighbors[num_neighbors].rating_n = training_data[j * ELEMS_IN_ROW + 3];
        num_neighbors++;
      }
    }
    // Add dummy nieghbor
    neighbors[num_neighbors].m_avg = m_sums[movie_m] / TRAINING_NUM_ROWS;
    neighbors[num_neighbors].nAvg = 0.;
    neighbors[num_neighbors].rRaw = 0.;
    neighbors[num_neighbors].weight = log(MIN_CV);
    neighbors[num_neighbors].rating_n = 0.;
    num_neighbors++;
    int max_neighbors = min(MAX_NEIGHBORS, num_neighbors);

    // Sort neighbors by weight, highest weight
    // int weight_compare(MovieNeighbor * a, MovieNeighbor * b) {
    qsort(neighbors, num_neighbors, sizeof(struct MovieNeighbor), weight_compare);

    float numerator = 0.; // sum(pred * weight)
    float denominator = 0.; // sum(wieght)
    int kkk;
    for (kkk = 0; kkk < max_neighbors; kkk++) {
      float dif = neighbors[kkk].rating_n - neighbors[kkk].nAvg;
      if (neighbors[kkk].rRaw < 0) {
        dif = -dif;
      }
      float prediction = neighbors[kkk].m_avg + dif;

      numerator += prediction * neighbors[kkk].weight;
      denominator += neighbors[kkk].weight;
    }

    float rating_pred = numerator / denominator;
    rating_pred = round_pos_float(rating_pred);

    //write to test_data_output
    test_data_output[i] = rating_pred;
    free(neighbors);
  }

  printf("\n===========\n");
}

void write_output_to_file(const float * test_data_output) {

  printf("Writing the predictions to: %s\n", TEST_OUTPUT_PATH);

  FILE * output_file;
  output_file = fopen(TEST_OUTPUT_PATH, "w");

  int hail_satan;
  for(hail_satan = 0; hail_satan < TEST_NUM_ROWS; hail_satan++) {
    if(hail_satan % 10000 == 0) {
      printf("\r%f writing predictions to file", ((float)hail_satan) / ((float)TEST_NUM_ROWS) );
    }
    char buffer [20];
    sprintf(buffer, "%.3f\n", test_data_output[hail_satan]);
    fputs(buffer, output_file);
  }

  printf("\nwrote %d predictions to file\n", hail_satan);

  fclose(output_file);
}

int main() {
   ///// Allocate  Memory ///////
   // training_data[row rum][col num], cols are (user, movie, date, rating)
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
    pearson_intermediates[l].r_raw = 0.;
    pearson_intermediates[l].weight = 0.;
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


  load_training(training_data);


#if COMPUTE_OR_LOAD == 1
  printf("COMPUTING PEARSON \n");
  compute_pearson_intermediate(training_data, user_start_idxs, m_sums, pearson_intermediates);
  write_pearson_intermediate(user_start_idxs, m_sums, pearson_intermediates);
#elif COMPUTE_OR_LOAD == 0
  printf("LOADING PEARSON \n");
  load_pearson_intermediate(user_start_idxs, m_sums, pearson_intermediates);
#endif
  ////////////////////////////////
  //// End of "training" part ////
  ////////////////////////////////
  float * test_data_output = (float*)malloc(TEST_NUM_ROWS * sizeof(float)); // TODO free

  // Load test set input
  int * test_data_input = (int*)malloc((TEST_NUM_ROWS + 1) * ELEMS_IN_TEST_ROW * sizeof(int)); // TODO free
  printf("allocated %f GB array test_data_input\n", (TEST_NUM_ROWS * ELEMS_IN_TEST_ROW * sizeof(int))/1000000000.);

  load_test(test_data_input);

  predict_test_set(m_sums, test_data_input, user_start_idxs, training_data, pearson_intermediates, test_data_output);


  // int jesus;
  // for( jesus = 0; jesus < 100; jesus++) {
  //   printf("%f\n", test_data_output[jesus]);
  // }

  write_output_to_file(test_data_output);


  printf("####################\n");printf("####################\n");printf("####################\n");printf("####################\n");printf("####################\n");printf("####################\n");printf("####################\n");
}
