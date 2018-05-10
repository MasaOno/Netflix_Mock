/*
https://classes.soe.ucsc.edu/cmps242/Fall09/proj/mpercy_svd_talk.pdf
http://sifter.org/~simon/journal/20061211.html
https://classes.soe.ucsc.edu/cmps242/Fall09/proj/mpercy_svd_paper.pdf


*/

// submit code: pdnijpwr

// g++ -o SVD_cpp fuckin_SVD.cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unordered_map>
#include <tuple>
#include <utility>      // std::pair, std::make_pair

// SVD parameters
#define L_RATE 0.001
#define NUM_FEATURES 40
#define EPOCHS 40
/*
#define L_RATE 0.001
#define NUM_FEATURES 10
#define EPOCHS 1
-> 3.8

#define L_RATE 0.001
#define NUM_FEATURES 10
#define EPOCHS 10
->

#define L_RATE 0.001
#define NUM_FEATURES 40
#define EPOCHS 100


*/

// Constants
#define NUM_USERS 458293 // 1 indexed
#define NUM_MOVIES 17770 // 1 indexed
#define ELEMS_IN_ROW 4 // 4 elements in a row: user, movie, date, rating
#define ELEMS_IN_TEST_ROW 3 // 4 elements in a row: user, movie, date

#define TRAINING_DATA_VERSION 1

// All data is in UM form
#if TRAINING_DATA_VERSION == 1 // Full UM dataset on Ethan's machine
// TODO: path here
  #define TRAINING_DATA_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt"
  #define TRAINING_NUM_ROWS 94362233
#endif



  /* CHOOSE SET TO PREDICT ON  */

  // // For test submission
  // #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um/qual.dta"
  // #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_subToServer_output.txt"
  // #define TEST_NUM_ROWS 2749898

  // // For validation set: index 2
  // #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/valid2.txt"
  // #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_VALIDATION.txt"
  // #define TEST_NUM_ROWS 1965045

  // // For hidden set, idx3

  // // For probe set, idx4
  // TODO: path here
  #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/probe4.txt"
  // TODO: path here
  #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/SVD_elo_UM_output_PROBE.txt"
  #define TEST_NUM_ROWS 1374739

  // // For training error, RMSE
  // #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/um_other/training1.txt"
  // #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_output_TRAINING_PREDICTION.txt"
  // #define TEST_NUM_ROWS 94362233


  // // For Logic2 Training error
  // #define TEST_INPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/UM_logic_test2.txt"
  // #define TEST_OUTPUT_PATH "/Users/ethanlo1/Documents/16th/3rd_term/CS156/Netflix_Mock/data/KNN_UM_logic2_testErr.txt"
  // #define TEST_NUM_ROWS 20


  /* ~~END~~ CHOOSE SET TO PREDICT ON ~~END~~ */

typedef std::pair<int,int> pair;
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

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
        // training_data[line_num * ELEMS_IN_TEST_ROW + i] = atoi(num_buf);
        test_data_input[line_num * ELEMS_IN_TEST_ROW + i] = atoi(num_buf);
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

/*
  For training. saves previously computed.

  curFeatToComp: idx of feature to be computed . curFeatToComp-1 is the most recent feature computed
*/
double predictRatingTrain(int userNumber, int movieNumber, const double * userFeature, const double * movieFeature,
                      double * cumSum, int curFeatToComp,
                      int curTrainingRow) {
    int f;

    // past feature have been multiplyed in the past:
    //    userFeature[f * (NUM_USERS + 1) + userNumber] * movieFeature[f * (NUM_MOVIES + 1) + movieNumber]
    // where f < curFeatToComp ---->>>> also corresponds curTrainingRow
    // cumulative sum stored in cumSum(user, movie)
        // cumsum.insert(std::make_pair(user, movie), 0.);
        // cumSum.count(std::make_pair(user, movie));
        // cumsum[std::make_pair(userNumber, movieNumber)] += 6969.
    double sum = cumSum[curTrainingRow];

    for (f = curFeatToComp; f < NUM_FEATURES; f++) {
      sum += userFeature[f * (NUM_USERS + 1) + userNumber] * movieFeature[f * (NUM_MOVIES + 1) + movieNumber];
    }

    return sum;
}

/*
Prediction function for test output. does NOT save previously computed
*/
double predictRatingTest(int userNumber, int movieNumber, const double * userFeature, const double * movieFeature) {
    double sum = 0.;
    int f;

    for (f = 0; f < NUM_FEATURES; f++) {
      sum += userFeature[f * (NUM_USERS + 1) + userNumber] * movieFeature[f * (NUM_MOVIES + 1) + movieNumber];
    }

    return sum;
}

// /*
// f: feature to be trained
// */
// static inline
// void train(int userNumber, int movieNumber, double rating, double * userFeature, double * movieFeature, int f, double * cumSUm) {
//   // USUALLY increments by userNumber each call.
//
// 	double err = L_RATE * (rating - predictRating(movieNumber, userNumber, userFeature, movieFeature, cumSum));
//
//   double tmpUV = userFeature[f * (NUM_USERS + 1) + userNumber];
//
//   userFeature[f * (NUM_USERS + 1) + userNumber] += err * movieFeature[f * (NUM_MOVIES + 1) + movieNumber];
// 	movieFeature[f * (NUM_MOVIES + 1) + movieNumber] += err * tmpUV;
// }


int main() {
  // User and Movie featuer matrices. Both accessed with arr[feature][user/movie]
  double * userFeature = new double[NUM_FEATURES * (NUM_USERS + 1)]; // userFeature[feature * (NUM_USERS + 1) + userNumber]
  double * movieFeature = new double[NUM_FEATURES * (NUM_MOVIES + 1)]; // movieFeature[feature * (NUM_MOVIES + 1) + movieNumber]

  int i;
  for (i = 0; i < NUM_FEATURES * (NUM_USERS + 1); i++) {
    userFeature[i] = 0.1;
  }
  for (i = 0; i < NUM_FEATURES * (NUM_MOVIES + 1); i++) {
    movieFeature[i] = 0.1;
  }

  printf("userFeature and movieFeature allocated\n");

  // Training dataset: [row rum][col num], cols are (user, movie, date, rating)
  int * trainingData = new int[TRAINING_NUM_ROWS * ELEMS_IN_ROW]; // trainingData[row * ELEMS_IN_ROW + col]
  load_training(trainingData);

  // Test/validation data
  int * testDataInput = new int[(TEST_NUM_ROWS + 1) * ELEMS_IN_TEST_ROW]; // testDataInput[row * ELEMS_IN_TEST_ROW + col]
  load_test(testDataInput);

  // output
  float * testDataOutput = new float[TEST_NUM_ROWS];


  // store comtributions
  printf("Allocating space for cumSum...\n");
  double * cumSum = new double[TRAINING_NUM_ROWS];
  // std::unordered_map<pair, double, pair_hash> * cumSum = new std::unordered_map<pair, double, pair_hash>();
  // cumSum->reserve(TRAINING_NUM_ROWS);

  for (i = 0; i < TRAINING_NUM_ROWS; i++) {
    if(i % 10000 == 0) {
      printf("\r%f  cumSum zeroed out", ((float)i/ TRAINING_NUM_ROWS));
    }
    cumSum[i] = 0.;
  }
  printf("\ncumSum initialized and set to 0\n");


  int f,ep;
  for (f = 0; f < NUM_FEATURES; f++) {
    for(ep = 0; ep < EPOCHS; ep++) {
      printf("feature %d / %d. epoch %d / %d\n", f, NUM_FEATURES, ep, EPOCHS);

      for (i = 0; i < TRAINING_NUM_ROWS; i++) {
        // Training feature f on the training row[i]

        // Row to train on
        int userNumber = trainingData[i * ELEMS_IN_ROW + 0];
        int movieNumber = trainingData[i * ELEMS_IN_ROW + 1];
        double rating = (double)(trainingData[i * ELEMS_IN_ROW + 3]);


        // predictRatingTrain(int userNumber, int movieNumber, const double * userFeature, const double * movieFeature, double * cumSum, int curFeatToComp)
        double err = L_RATE * (rating - predictRatingTrain(userNumber, movieNumber, userFeature, movieFeature, cumSum, f, i));

        double tmpUV = userFeature[f * (NUM_USERS + 1) + userNumber];

        userFeature[f * (NUM_USERS + 1) + userNumber] += err * movieFeature[f * (NUM_MOVIES + 1) + movieNumber];
        movieFeature[f * (NUM_MOVIES + 1) + movieNumber] += err * tmpUV;

        // train(userNumber, movieNumber, rating, userFeature, movieFeature, f);

      }
    }
  }

  // predict on testDataOutput
  for (i = 0; i < TEST_NUM_ROWS; i++) {
    int userNumber = testDataInput[i * ELEMS_IN_TEST_ROW + 0]; // user
    int movieNumber = testDataInput[i * ELEMS_IN_TEST_ROW + 1]; // movie

    testDataOutput[i] = predictRatingTest(userNumber, movieNumber, userFeature, movieFeature);
  }

  write_output_to_file(testDataOutput);

}
