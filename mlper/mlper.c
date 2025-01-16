#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mlper.h"

// Feature and target variable arrays (feature arrays are multi-dimensional)
double *X = NULL;
double *X_train = NULL;
double *X_test = NULL;

double *y = NULL;
double *y_train = NULL;
double *y_test = NULL;

// Predictions array
double *y_pred = NULL;

// Initialize length of data splits
int train_length = 0;
int test_length = 0;

// Initialize counts of features/entries
int num_features = 0;
int num_entries = 0;

// Dynamically initialize memory for global arrays to 0
void initialize_globals(int num_features, int num_entries)
{
    // For multidimensional feature arrays, allocate as a single memory block for efficiency
    X = calloc(num_entries * num_features, sizeof(double));
    X_train = calloc(num_entries * num_features, sizeof(double));
    X_test = calloc(num_entries * num_features, sizeof(double));

    y = calloc(num_entries, sizeof(double));
    y_train = calloc(num_entries, sizeof(double));
    y_test = calloc(num_entries, sizeof(double));

    y_pred = calloc(num_entries, sizeof(double));

    // Ensure memory was able to be allocated
    if (!X || !y || !X_train || !X_test || !y_train || !y_test)
    {
        printf("Unable to allocate memory.\n");
        exit(EXIT_FAILURE);
    }
}

// Free memory allocated for global arrays
void free_globals()
{
    free(X);
    free(y);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(y_pred);
}

// Load CSV at specified file path into feature and target variable arrays
void load(char *file_path)
{
    // Reading buffer
    char line[MAX_LENGTH];
    // Tokenized entry segment
    char *token;

    // First file read, counts features and entries
    FILE *file = fopen(file_path, "r");
    if (!file)
    {
        printf("Unable to open file at: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // Read header row into buffer
    fgets(line, MAX_LENGTH, file);

    // Tokenize header, splitting by commas
    token = strtok(line, ",");

    // Count number of features
    while (token)
    {
        num_features++;
        // Continue iterating tokens in line
        token = strtok(NULL, ",");
    }

    // Ensure feature count excludes target variable
    num_features--;

    // Iterate remaining lines in file, counting number of data entries
    while (fgets(line, MAX_LENGTH, file) != NULL)
    {
        num_entries++;
    }

    fclose(file);

    // Dynamically memory for data arrays
    initialize_globals(num_features, num_entries);

    // Second file read, copies data into allocated memory
    file = fopen(file_path, "r");
    if (!file)
    {
        printf("Unable to reopen file at: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // Skip header
    fgets(line, MAX_LENGTH, file);

    int index = 0;

    // Read CSV file by line
    while (fgets(line, MAX_LENGTH, file) != NULL)
    {
        // Tokenize row, splitting by commas
        token = strtok(line, ",");
        // Add feature variables
        for (int i = 0; i < num_features; i++)
        {
            // Cast to double and assign to matrix
            X[index * num_features + i] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        // Add target variable
        y[index] = strtod(token, NULL);

        index++;
    }

    fclose(file);
}

// Standardize feature data to mean of 0, standard deviation of 1
void standardize(double *X, int num_features, int num_entries)
{
    // Iterate input features
    for (int j = 0; j < num_features; j++)
    {
        // Sum feature data
        double sum = 0.0;
        for (int i = 0; i < num_entries; i++)
        {
            sum += X[i * num_features + j];
        }
        // Calculate mean
        double mean = sum / num_entries;

        // Sum differences from mean squared
        double sum_diff_squared = 0.0;
        for (int i = 0; i < num_entries; i++)
        {
            double diff = X[i * num_features + j] - mean;
            sum_diff_squared += diff * diff;
        }
        // Calculate standard deviation
        double std_dev = sqrt(sum_diff_squared / num_entries);

        // Perform feature standardization
        // X = (X - mean) / std_dev
        for (int i = 0; i < num_entries; i++)
        {
            X[i * num_features + j] = (X[i * num_features + j] - mean) / std_dev;
        }
    }
}

// Splits data into training/test sets given test proportion
void train_test_split(double test_proportion)
{
    // Optionally seed psuedo rng
    // srand(115);

    // Iterate indices of sample data
    for (int i = 0; i < num_entries; i++)
    {
        // Generate random number between 0 and 1
        float number = rand() / ((double) RAND_MAX + 1);

        // Assign to correct data split
        if (number <= test_proportion)
        {
            // Copy all feature values
            for (int j = 0; j < num_features; j++)
            {
                X_test[test_length * num_features + j] = X[i * num_features + j];
            }
            y_test[test_length] = y[i];
            test_length++;
        }
        else
        {
            // Copy all feature values
            for (int j = 0; j < num_features; j++)
            {
                X_train[train_length * num_features + j] = X[i * num_features + j];
            }
            y_train[train_length] = y[i];
            train_length++;
        }
    }

    // Dynamically reallocate (resize) memory to true array sizes
    X_train = realloc(X_train, train_length * num_features * sizeof(double));
    X_test = realloc(X_test, test_length * num_features * sizeof(double));
    y_train = realloc(y_train, train_length * sizeof(double));
    y_test = realloc(y_test, test_length * sizeof(double));
}

// Computes Mean Squared Error (MSE) of predicted values
double mean_squared_error(double *y_actual, double *y_pred, int num_predictions)
{
    // Sum of absolute error squared
    double sum_error = 0;
    // Iterate predictions
    for (int i = 0; i < num_predictions; i++)
    {
        // Calculate error
        double abs_error = y_actual[i] - y_pred[i];
        // Add squared result to sum of error
        sum_error += abs_error * abs_error;
    }

    // Return average sum of absolute error squared
    return (sum_error / num_predictions);
}

// Export prediction array to CSV
void export_predictions(char *file_name, double *y_pred, int num_predictions)
{
    // Edit existing CSV file or create new one
    FILE *file = fopen(file_name, "w");
    if (!file)
    {
        printf("Unable to open/create file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    // Write column header
    fprintf(file, "y\n");

    // Write predictions to output file
    for (int i = 0; i < num_predictions; i++)
    {
        fprintf(file, "%f\n", y_pred[i]);
    }
    fclose(file);
}

// Export complete test data and predicted values to CSV
void export_results(char *file_name, double *X, double *y_actual, double *y_pred, int num_features, int num_predictions)
{
    // Edit existing CSV file or create new one
    FILE *file = fopen(file_name, "w");
    if (!file)
    {
        printf("Unable to open/create file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    // Write column headers
    for (int j = 0; j < num_features; j++)
    {
        fprintf(file, "X%d,", j + 1);
    }
    fprintf(file, "y,y_pred\n");

    // Write rows of data
    for (int i = 0; i < num_predictions; i++)
    {
        // Write feature data
        for (int j = 0; j < num_features; j++)
        {
            fprintf(file, "%f,", X[i * num_features + j]);
        }
        // Write target data
        fprintf(file, "%f,%f\n", y_actual[i], y_pred[i]);
    }
    fclose(file);
}
