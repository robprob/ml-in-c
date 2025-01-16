#ifndef MLPER_H
#define MLPER_H

#include <stdio.h>

// Max char length of input file row
#define MAX_LENGTH sizeof(char) * 1000

// Feature and target variable arrays
extern double *X;
extern double *X_train;
extern double *X_test;

extern double *y;
extern double *y_train;
extern double *y_test;

// Predictions array
extern double *y_pred;

// Initialize length of data splits
extern int train_length;
extern int test_length;

// Initialize counts of features/entries
extern int num_features;
extern int num_entries;

// Function prototypes
void initialize_globals(int num_features, int num_entries);
void free_globals();
void load(char *file_path);
void standardize(double *X, int num_features, int num_entries);
void train_test_split(double test_proportion);
double mean_squared_error(double *y_actual, double *y_pred, int num_predictions);
void export_predictions(char *file_name, double *y_pred, int num_predictions);
void export_results(char *file_name, double *X, double *y_actual, double *y_pred, int num_features, int num_predictions);

#endif
