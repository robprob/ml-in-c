#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "dataset.h"
#include "metrics.h"


// Transforms input feature matrix into polynomial feature matrix of specified maximum degree
void poly_transform(struct Dataset *data, int degree)
{
    if (degree <= 1)
    {
        return;
    }

    double *X = data->X;
    int num_entries = data->num_entries;
    int num_features = data->num_features;

    // Allocate memory for new matrix
    int poly_num_features = num_features * degree;
    double *X_poly = calloc(num_entries * poly_num_features, sizeof(double));
    if (!X_poly)
    {
        printf("Unable to allocate memory for polynomial features.\n");
        exit(EXIT_FAILURE);
    }

    // Generate polynomial feature matrix
    // Layout example: Features = [X1, X2], Degree = 2 -> [X1, X1^2, X2, X2^2]
    for (int i = 0; i < num_entries; i++)
    {
        int current_col = 0;
        for (int j = 0; j < num_features; j++)
        {
            for (int power = 1; power <= degree; power++)
            {
                X_poly[i * poly_num_features + current_col] = double_pow(X[i * num_features + j], power);
                current_col++;
            }
        }
    }

    // Free original pointer and reassign
    free(X);
    data->X = X_poly;
    // Update feature count
    data->num_features = poly_num_features;
}

// Standardize feature data to mean of 0, standard deviation of 1, using training data statistics
void standardize(struct Dataset *data)
{
    if (!data->standardized)
    {
        return;
    }

    // Dynamically allocate memory for training feature statistics, initialized at 0
    data->feature_means = calloc(data->num_features, sizeof(double));
    data->feature_stds = calloc(data->num_features, sizeof(double));
    if (!data->feature_means || !data->feature_stds)
    {
        printf("Unable to allocate memory for feature statistics.\n");
        exit(EXIT_FAILURE);
    }

    // Iterate input features
    for (int j = 0; j < data->num_features; j++)
    {
        // Sum feature data of training only
        double sum = 0.0;
        for (int i = 0; i < data->train_length; i++)
        {
            sum += data->X_train[i * data->num_features + j];
        }
        // Calculate mean
        double mean = sum / data->train_length;
        // Save to means array
        data->feature_means[j] = mean;

        // Sum differences from mean squared
        double sum_diff_squared = 0.0;
        for (int i = 0; i < data->train_length; i++)
        {
            double diff = data->X_train[i * data->num_features + j] - mean;
            sum_diff_squared += diff * diff;
        }
        // Calculate standard deviation
        double std_dev = sqrt(sum_diff_squared / data->train_length);
        // Save to stds array
        data->feature_stds[j] = std_dev;

        // Standardize training data
        for (int i = 0; i < data->train_length; i++)
        {
            // X = (X - mean) / std_dev
            data->X_train[i * data->num_features + j] = (data->X_train[i * data->num_features + j] - mean) / std_dev;
        }
        // Standardize test data
        for (int i = 0; i < data->test_length; i++)
        {
            // X = (X - mean) / std_dev
            data->X_test[i * data->num_features + j] = (data->X_test[i * data->num_features + j] - mean) / std_dev;
        }
    }
}

// Un-standardize specified feature data matrix back to original values
void unstandardize(struct Dataset *data, double *feature_data, int num_entries)
{
    // Iterate data entries
    for (int i = 0; i < num_entries; i++)
    {
        // Iterate data features
        for (int j = 0; j < data->num_features; j++)
        {
            // Perform feature un-standardization
            // X = (X - mean) / std_dev
            // X = (X * std_dev) + mean
            feature_data[i * data->num_features + j] = (feature_data[i * data->num_features + j] * data->feature_stds[j]) + data->feature_means[j];
        }
    }
}
