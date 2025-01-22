#ifndef MLPER_H
#define MLPER_H

#include <stdio.h>

struct Dataset {
    char file_path[128];     // Path to CSV file
    int standardized;        // Standardization truth value
    double *feature_means;   // Mean of each input feature
    double *feature_stds;    // Standard deviation of each input feature
    double *X;               // Feature matrix
    double *y;               // Target array
    double *X_train;         // Training feature set
    double *X_valid;         // Validation feature set
    double *X_test;          // Test feature set
    double *y_train;         // Training targets
    double *y_valid;         // Validation targets
    double *y_test;          // Test targets
    double *y_pred;          // Model Predictions
    int num_features;        // Number of features
    int num_entries;         // Number of entries
    double test_proportion;  // Proportion of training data held in test set
    double valid_proportion; // Proportion of training set split into validation set
    int train_length;        // Number of training entries
    int valid_length;        // Number of validation entries
    int test_length;         // Number of test entries
};

// Function prototypes
void initialize_dataset(struct Dataset *data, int num_features, int num_entries);
void free_dataset(struct Dataset *data);
void load(struct Dataset *data);
void standardize(struct Dataset *data);
void unstandardize(struct Dataset *data, double *feature_data, int num_entries);
void train_test_split(struct Dataset *data, double test_proportion);
void validation_split(struct Dataset *data, double valid_proportion);
void shuffle_batch(struct Dataset *data, int batch_size);
double mean_squared_error(double *y_actual, double *y_pred, int num_predictions);
void export_predictions(double *y_pred, int num_predictions, char *file_name);
void export_results(struct Dataset *data, int num_predictions, char *file_name);

#endif
