#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

// Split data into training/test sets
void train_test_split(struct Dataset *data, double test_proportion)
{

    int num_entries = data->num_entries;
    int num_features = data->num_features;

    int train_length = 0;
    int test_length = 0;


    // Seed psuedo rng with current time
    srand(time(NULL));

    // Iterate indices of sample data
    for (int i = 0; i < num_entries; i++)
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

// Shuffle random entries to "generate" a training batch of specified size
void shuffle_batch(struct Dataset *data, int batch_size)
{
    int train_length = data->train_length;
    int num_features = data->num_features;
    double *X_train = data->X_train;
    double *y_train = data->y_train;

    // Swap random training entries into the first i positions of the dataset, where i is batch size
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

// Computes Mean Squared Error (MSE) between predicted and actual values
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
    double *X_test = data->X_test;
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
            fprintf(file, "%f,", X_test[i * num_features + j]);
        }
        // Write target data
        fprintf(file, "%f,%f\n", y_actual[i], y_pred[i]);
    }
    fclose(file);
}
