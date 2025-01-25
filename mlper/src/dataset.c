#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "dataset.h"


// Dynamically initialize memory for dataset to 0
void initialize_dataset(struct Dataset *data, int num_features, int num_samples)
{
    data->X = calloc(num_samples * num_features, sizeof(double));
    data->y = calloc(num_samples, sizeof(double));

    if (!data->X || !data->y) {
        printf("Unable to allocate memory for dataset.\n");
        exit(EXIT_FAILURE);
    }
}

// Dynamically initialize memory for training and test splits to 0
void initialize_splits(struct Dataset *data, int num_features, int num_samples)
{
    data->X_train = calloc(num_samples * num_features, sizeof(double));
    data->X_test = calloc(num_samples * num_features, sizeof(double));
    data->y_train = calloc(num_samples, sizeof(double));
    data->y_test = calloc(num_samples, sizeof(double));

    if (!data->X_train || !data->X_test || !data->y_train || !data->y_test) {
        printf("Unable to allocate memory for dataset splits.\n");
        exit(EXIT_FAILURE);
    }
}

// Free memory allocated for all dataset arrays
void free_dataset(struct Dataset *data)
{
    free(data->X);
    free(data->y);
    free(data->X_train);
    free(data->X_valid);
    free(data->X_test);
    free(data->y_train);
    free(data->y_valid);
    free(data->y_test);
    free(data->y_pred);

    // Free feature statistics, if applicable
    free(data->feature_means);
    free(data->feature_stds);
}

// Split data into training/test sets
void train_test_split(struct Dataset *data, double test_proportion)
{

    int num_samples = data->num_samples;
    int num_features = data->num_features;

    int train_length = 0;
    int test_length = 0;


    // Seed psuedo rng with current time
    srand(time(NULL));

    // Iterate indices of sample data
    for (int i = 0; i < num_samples; i++)
    {
        // Generate random number between 0 and 1
        float number = (float) rand() / RAND_MAX;

        // Assign to correct data split
        if (number <= test_proportion)
        {
            // Copy all feature values
            for (int j = 0; j < num_features; j++)
            {
                data->X_test[test_length * num_features + j] = data->X[i * num_features + j];
            }
            data->y_test[test_length] = data->y[i];
            test_length++;
        }
        else
        {
            // Copy all feature values
            for (int j = 0; j < num_features; j++)
            {
                data->X_train[train_length * num_features + j] = data->X[i * num_features + j];
            }
            data->y_train[train_length] = data->y[i];
            train_length++;
        }
    }

    // Assign updated array lengths
    data->train_length = train_length;
    data->test_length = test_length;

    // Dynamically reallocate memory to true array sizes
    data->X_train = realloc(data->X_train, data->train_length * data->num_features * sizeof(double));
    data->X_test = realloc(data->X_test, data->test_length * data->num_features * sizeof(double));
    data->y_train = realloc(data->y_train, data->train_length * sizeof(double));
    data->y_test = realloc(data->y_test, data->test_length * sizeof(double));
}

// Further split training data into training/validation set
void validation_split(struct Dataset *data, double valid_proportion)
{
    if (valid_proportion == 0.0)
    {
        return;
    }

    // Seed psuedo rng with current time
    srand(time(NULL));

    int train_length = data->train_length;
    int new_train_length = 0;

    int valid_length = 0;

    int num_features = data->num_features;

    // Initialize memory for validation arrays to 0
    data->X_valid = calloc(train_length * num_features, sizeof(double));
    data->y_valid = calloc(train_length, sizeof(double));
    if (!data->X_valid || !data->y_valid)
    {
        printf("Unable to allocate memory for validation set.\n");
        exit(EXIT_FAILURE);
    }

    // Adjust valid proportion, which is proportion of total dataset, to training set size
    double train_proportion = (1 - data->test_proportion);
    double adjusted_valid_proportion = (valid_proportion / train_proportion);

    // Iterate training set
    for (int i = 0; i < train_length; i++)
    {
        // Generate random number between 0 and 1
        float number = (float) rand() / RAND_MAX;

        // Assign to correct data split
        if (number <= adjusted_valid_proportion)
        {
            // Iterate feature columns, copying values to validation set
            for (int j = 0; j < num_features; j++)
            {
                data->X_valid[valid_length * num_features + j] = data->X_train[i * num_features + j];
            }
            // Copy target values
            data->y_valid[valid_length] = data->y_train[i];
            valid_length++;
        }
        else
        {
            // Iterate feature columns, copying values back to new location in train set
            for (int j = 0; j < num_features; j++)
            {
                data->X_train[new_train_length * num_features + j] = data->X_train[i * num_features + j];
            }
            // Copy target values
            data->y_train[new_train_length] = data->y_train[i];
            new_train_length++;
        }
    }

    // Assign updated array lengths
    data->valid_length = valid_length;
    data->train_length = new_train_length;

    // Dynamically reallocate memory to true array sizes
    // Using realloc with updated (reduced) training length effectively trims the extra data
    data->X_train = realloc(data->X_train, data->train_length * data->num_features * sizeof(double));
    data->X_valid = realloc(data->X_valid, data->valid_length * data->num_features * sizeof(double));
    data->y_train = realloc(data->y_train, data->train_length * sizeof(double));
    data->y_valid = realloc(data->y_valid, data->valid_length * sizeof(double));
}

// Shuffle random samples to "generate" a training batch of specified size
void shuffle_batch(struct Dataset *data, int batch_size)
{
    int train_length = data->train_length;
    int num_features = data->num_features;
    double *X_train = data->X_train;
    double *y_train = data->y_train;

    // Swap random training samples into the first i positions of the dataset, where i is batch size
    for (int i = 0; i < batch_size; i++)
    {
        // Generate random index in bounds of training set
        int rand_index = rand() % train_length;
        // Iterate feature values
        for (int j = 0; j < num_features; j++)
        {
            // Swap feature data on randomly selected index
            double temp_X = X_train[i * num_features + j];
            X_train[i * num_features + j] = X_train[rand_index * num_features + j];
            X_train[rand_index * num_features + j] = temp_X;
        }
        // Swap target variable data on randomly selected index
        double temp_y = y_train[i];
        y_train[i] = y_train[rand_index];
        y_train[rand_index] = temp_y;
    }
}
