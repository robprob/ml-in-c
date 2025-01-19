#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mlper.h"

// Dynamically initialize memory for global arrays to 0
void initialize_dataset(struct Dataset *data, int num_features, int num_entries)
{
    data->X = calloc(num_entries * num_features, sizeof(double));
    data->y = calloc(num_entries, sizeof(double));
    data->X_train = calloc(num_entries * num_features, sizeof(double));
    data->X_test = calloc(num_entries * num_features, sizeof(double));
    data->y_train = calloc(num_entries, sizeof(double));
    data->y_test = calloc(num_entries, sizeof(double));

    if (!data->X || !data->y || !data->X_train || !data->X_test || !data->y_train || !data->y_test) {
        printf("Unable to allocate memory for dataset.\n");
        exit(EXIT_FAILURE);
    }
}

// Free memory allocated for global arrays
void free_dataset(struct Dataset *data)
{
    free(data->X);
    free(data->y);
    free(data->X_train);
    free(data->X_test);
    free(data->y_train);
    free(data->y_test);
    free(data->y_pred);
}

// Load CSV at specified file path into feature and target variable arrays
void load(struct Dataset *data)
{
    // First file read: determine proper header length, count features/entries
    FILE *file = fopen(data->file_path, "r");
    if (!file)
    {
        printf("Unable to open file at: %s\n", data->file_path);
        exit(EXIT_FAILURE);
    }

    // Start with modestly-large buffer
    size_t header_size = 1024;
    char *header = malloc(header_size);
    if (!header)
    {
        printf("Unable to allocate memory for header.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t bytes_read = 0;
    // Continually read
    while (fgets(header + bytes_read, header_size - bytes_read, file))
    {
        // If header contains line break, full header has been read
        if (strchr(header + bytes_read, '\n'))
        {
            break;
        }
        // Double buffer size, reallocating memory
        // Provides flexibility for high-dimensional datasets
        header_size *= 2;
        header = realloc(header, header_size);
        // Catch invalid or extremely large header size allocations
        if (!header || header_size > 1000000)
        {
            printf("Unable to reallocate memory for larger header.\n");
            free(header);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        // Update total bytes read
        bytes_read = strlen(header);
    }

    // Tokenize header, splitting by commas
    char *token = strtok(header, ",");

    // Count number of features
    int num_features = 0;
    while (token)
    {
        num_features++;
        // Continue iterating tokens in header
        token = strtok(NULL, ",");
    }
    free(header);
    // Exclude target variable from feature count
    data->num_features = num_features - 1;

    // Given feature count, forgivingly over-estimate maximum byte length of data entries
    int max_float_length = 20; // over-estimate 20 character float max
    int max_length = (data->num_features + 1) * (max_float_length + 1); // + 1 accommodates commas, \n, etc.
    char *line = malloc(max_length);
    if (!line)
    {
        printf("Unable to allocate memory for data rows.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Iterate remaining lines in file, counting number of data entries
    while (fgets(line, max_length, file) != NULL)
    {
        data->num_entries++;
    }
    fclose(file);

    // Dynamically allocate memory for dataset arrays
    initialize_dataset(data, data->num_features, data->num_entries);

    // Second file read: copy data into allocated memory
    file = fopen(data->file_path, "r");
    if (!file)
    {
        printf("Unable to reopen file at: %s\n", data->file_path);
        exit(EXIT_FAILURE);
    }

    // Skip header
    fgets(line, max_length, file);

    // Read CSV, file by line
    int index = 0;
    while (fgets(line, max_length, file) != NULL)
    {
        // Tokenize row, splitting by commas
        token = strtok(line, ",");
        // Write feature variables to dataset
        for (int i = 0; i < data->num_features; i++)
        {
            // Cast to double and assign to feature matrix
            data->X[index * data->num_features + i] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        // Add target variable
        data->y[index] = strtod(token, NULL);

        index++;
    }
    free(line);
    fclose(file);
}

// Standardize feature data to mean of 0, standard deviation of 1
void standardize(struct Dataset *data)
{
    // Iterate input features
    for (int j = 0; j < data->num_features; j++)
    {
        // Sum feature data
        double sum = 0.0;
        for (int i = 0; i < data->num_entries; i++)
        {
            sum += data->X[i * data->num_features + j];
        }
        // Calculate mean
        double mean = sum / data->num_entries;

        // Sum differences from mean squared
        double sum_diff_squared = 0.0;
        for (int i = 0; i < data->num_entries; i++)
        {
            double diff = data->X[i * data->num_features + j] - mean;
            sum_diff_squared += diff * diff;
        }
        // Calculate standard deviation
        double std_dev = sqrt(sum_diff_squared / data->num_entries);

        // Perform feature standardization
        // X = (X - mean) / std_dev
        for (int i = 0; i < data->num_entries; i++)
        {
            data->X[i * data->num_features + j] = (data->X[i * data->num_features + j] - mean) / std_dev;
        }
    }
}

// Split data into training/test sets
void train_test_split(struct Dataset *data, double test_proportion)
{
    // Optionally seed psuedo rng
    // srand(115);

    // Iterate indices of sample data
    for (int i = 0; i < data->num_entries; i++)
    {
        // Generate random number between 0 and 1
        float number = rand() / ((double) RAND_MAX + 1);

        // Assign to correct data split
        if (number <= test_proportion)
        {
            // Copy all feature values
            for (int j = 0; j < data->num_features; j++)
            {
                data->X_test[data->test_length * data->num_features + j] = data->X[i * data->num_features + j];
            }
            data->y_test[data->test_length] = data->y[i];
            data->test_length++;
        }
        else
        {
            // Copy all feature values
            for (int j = 0; j < data->num_features; j++)
            {
                data->X_train[data->train_length * data->num_features + j] = data->X[i * data->num_features + j];
            }
            data->y_train[data->train_length] = data->y[i];
            data->train_length++;
        }
    }

    // Dynamically reallocate memory to true array sizes
    data->X_train = realloc(data->X_train, data->train_length * data->num_features * sizeof(double));
    data->X_test = realloc(data->X_test, data->test_length * data->num_features * sizeof(double));
    data->y_train = realloc(data->y_train, data->train_length * sizeof(double));
    data->y_test = realloc(data->y_test, data->test_length * sizeof(double));
}

// Computes Mean Squared Error (MSE) between predicted and actual values.
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

// Exports predictions to a CSV file
void export_predictions(double *y_pred, int num_predictions, char *file_name)
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

// Exports full data, actual values, and predictions to a CSV file
void export_results(struct Dataset *data, int num_predictions, char *file_name)
{
    double *X = data->X;
    double *y_actual = data->y_test;
    double *y_pred = data->y_pred;
    int num_features = data->num_features;

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
