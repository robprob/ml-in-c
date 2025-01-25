#ifndef DATASET_H
#define DATASET_H

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
    int num_samples;         // Number of samples
    double test_proportion;  // Proportion of training data held in test set
    double valid_proportion; // Proportion of training set split into validation set
    int train_length;        // Number of training samples
    int valid_length;        // Number of validation samples
    int test_length;         // Number of test samples
};

// Memory management
void initialize_dataset(struct Dataset *data, int num_features, int num_samples);
void initialize_splits(struct Dataset *data, int num_features, int num_samples);
void free_dataset(struct Dataset *data);

// Dataset splitting
void train_test_split(struct Dataset *data, double test_proportion);
void validation_split(struct Dataset *data, double valid_proportion);

// Generate stochastic batches
void shuffle_batch(struct Dataset *data, int batch_size);

#endif
