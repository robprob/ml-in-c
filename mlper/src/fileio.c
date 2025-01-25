#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "preprocessing.h"


// Load CSV at specified file path into feature and target variable arrays
void load(struct Dataset *data)
{
    // First file read: determine proper header length, count features/samples
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

    // Given feature count, forgivingly over-estimate maximum byte length of data samples
    int max_float_length = 20; // over-estimate 20 character float max
    int max_length = (data->num_features + 1) * (max_float_length + 1); // + 1 accommodates commas, \n, etc.
    char *line = malloc(max_length);
    if (!line)
    {
        printf("Unable to allocate memory for data rows.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Iterate remaining lines in file, counting number of data samples
    while (fgets(line, max_length, file) != NULL)
    {
        data->num_samples++;
    }
    fclose(file);

    // Dynamically allocate memory for dataset arrays
    initialize_dataset(data, data->num_features, data->num_samples);

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
void export_results(struct Dataset *data, int num_predictions, int poly_degrees, char *file_name)
{
    double *X_test = data->X_test;
    double *y_actual = data->y_test;
    double *y_pred = data->y_pred;
    int num_features = data->num_features;

    // Un-standardize input matrix, if necessary
    if (data->standardized)
    {
        unstandardize(data, data->X_test, data->test_length);
    }

    // Edit existing CSV file or create new one
    FILE *file = fopen(file_name, "w");
    if (!file)
    {
        printf("Unable to open/create file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    int og_num_features = num_features / poly_degrees;

    // Write column headers
    for (int j = 1; j <= og_num_features; j++)
    {
        for (int d = 1; d <= poly_degrees; d++)
        {
            fprintf(file, "X%i^%i,", j, d);
        }
    }
    fprintf(file, "y,y_pred\n");

    // Write rows of data
    for (int i = 0; i < num_predictions; i++)
    {
        // Write feature data
        for (int j = 0; j < num_features; j++)
        {
            fprintf(file, "%f,", X_test[i * num_features + j]);
        }
        // Write target data
        fprintf(file, "%f,%f\n", y_actual[i], y_pred[i]);
    }
    fclose(file);
}
