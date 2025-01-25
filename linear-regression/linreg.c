/*
Implementation of multi-variable linear regression.

Linear Function
h(x) = wi * xj + b

h(x): linear prediction
wi: weight vector
xj: feature vector
b: bias parameter

Mean Squared Error (MSE) Function
E = (1/2) * Σ(i=1)^n (yi - ^yi)^2
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "mlper.h"

struct LinearRegression {
    int num_epochs;            // Number of training iterations
    int polynomial_degree;     // Highest polynomial degree to generate for Polynomial Regression
    double learning_rate;      // Training "step" size
    int early_stopping;        // Truth value of early stopping
    int batch_size;            // Number of samples taken per epoch (leave at 0 for batch GD)
    double l2_alpha;           // Ridge coefficient
    double l1_alpha;           // Lasso coefficient
    double mix_ratio;          // 0 is equivalent to pure ridge, 1 is equivalent to pure lasso
    double *w;                 // Weight vector
    double b;                  // Bias
};

// Function Prototypes
void parse_config(struct Dataset *data, struct LinearRegression *linreg);
void initialize_model(struct LinearRegression *linreg, int num_features);
void fit_model(struct LinearRegression *linreg, struct Dataset *data);
void print_model_parameters(struct LinearRegression *linreg, struct Dataset *data);
void predict(struct LinearRegression *linreg, struct Dataset *data, double *X_predict, int num_predictions);


int main(int argc, char **argv)
{
    clock_t start, end;

    start = clock();

    // Instantiate dataset at 0
    struct Dataset data = {0};
    // Instantiate Linear Regression model at 0
    struct LinearRegression linreg = {0};

    // Parse data path and model hyperparameter from config file
    parse_config(&data, &linreg);

    /*
    // Print selected parameters
    printf("File Path: %s\n", data.file_path);
    printf("Standardized: %i\n", data.standardized);
    printf("Number of Epochs: %i\n", linreg.num_epochs);
    printf("Learning Rate: %g\n", linreg.learning_rate);
    printf("Test Proportion: %g\n", data.test_proportion);
    printf("Validation Proportion: %g\n", data.valid_proportion);
    printf("Early Stopping: %i\n", linreg.early_stopping);
    printf("Batch Size: %i\n", linreg.batch_size);
    printf("L2 alpha: %g\n", linreg.l2_alpha);
    printf("L1 alpha: %g\n", linreg.l1_alpha);
    printf("Elastic Net Mix Ratio: %g\n", linreg.mix_ratio);
    */

    // Load feature and target variable data into arrays
    load(&data);
    // Transform feature matrix to specified highest degree
    poly_transform(&data, linreg.polynomial_degree);
    // Dynamically allocate memory for train/test splits based on any feature transformation
    initialize_splits(&data, data.num_features, data.num_samples);
    // Split data into training and test sets
    train_test_split(&data, data.test_proportion);
    // If specified, standardize feature arrays to mean of 0, standard deviation of 1
    standardize(&data);
    // Further split training data into validation set, if specified
    validation_split(&data, data.valid_proportion);

    // Print size of each data set
    printf("Training Size: %i\n", data.train_length);
    printf("Validation Size: %i\n", data.valid_length);
    printf("Test Size: %i\n\n", data.test_length);

    // Initialize memory for model parameters at 0
    initialize_model(&linreg, data.num_features);

    clock_t train_start, train_end;

    train_start = clock();

    // Fit model to training data
    fit_model(&linreg, &data);

    train_end = clock();

    // Print trained model parameters, un-standardized if necessary
    print_model_parameters(&linreg, &data);

    // Generate predictions using trained model
    predict(&linreg, &data, data.X_test, data.test_length);

    // Calculate Mean Squared Error and RMSE
    double mse = mean_squared_error(data.y_test, data.y_pred, data.test_length);
    printf("\nTest MSE: %f\n", mse);
    printf("Test RMSE: %f\n", sqrt(mse));

    // Calculate and print model training time
    double training_time = ((double) (train_end - train_start)) / CLOCKS_PER_SEC;
    printf("\nTraining Time: %f seconds\n", training_time);

    // Export feature data and calculated predictions
    export_results(&data, data.test_length, linreg.polynomial_degree, "test_predictions.csv");

    // Free memory taken up by dataset
    free_dataset(&data);
    // Free parameter weights array
    free(linreg.w);

    end = clock();
    double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Total CPU Time: %f seconds\n", total_time);
}

// Parse data file path and model hyperparameter from config file
void parse_config(struct Dataset *data, struct LinearRegression *linreg)
{
    FILE *file = fopen("config.txt", "r");
    if (!file)
    {
        printf("Unable to open config file, config.txt\n");
        exit(EXIT_FAILURE);
    }

    // Line reading buffer
    char line[256];

    while (fgets(line, sizeof(line), file))
    {
        char key [128];
        char value[128];

        // Skip headers and blank lines
        if (line[0] == '[' || line[0] == '\n')
        {
            continue;
        }

        if (sscanf(line, "%[^=]=%s", key, value) == 2)
        {
            if (strcmp(key, "file_path ") == 0)
            {
                strcpy(data->file_path, value);
            }
            else if (strcmp(key, "polynomial_degree ") == 0)
            {
                linreg->polynomial_degree = atoi(value);
            }
            else if (strcmp(key, "standardize ") == 0)
            {
                data->standardized = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "num_epochs ") == 0)
            {
                linreg->num_epochs = atoi(value);
            }
            else if (strcmp(key, "learning_rate ") == 0)
            {
                linreg->learning_rate = atof(value);
            }
            else if (strcmp(key, "test_proportion ") == 0)
            {
                data->test_proportion = atof(value);
            }
            else if (strcmp(key, "valid_proportion ") == 0)
            {
                data->valid_proportion = atof(value);
            }
            else if (strcmp(key, "early_stopping ") == 0)
            {
                linreg->early_stopping = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "batch_size ") == 0)
            {
                linreg->batch_size = atoi(value);
            }
            else if (strcmp(key, "l2_alpha ") == 0)
            {
                linreg->l2_alpha = atof(value);
            }
            else if (strcmp(key, "l1_alpha ") == 0)
            {
                linreg->l1_alpha = atof(value);
            }
            else if (strcmp(key, "mix_ratio ") == 0)
            {
                linreg->mix_ratio = atof(value);
            }
            else
            {
                printf("Invalid config key, %s\n", key);
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            printf("Invalid config line: %s\n", line);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

// Initialize model paramaters at 0 (horizontal line)
void initialize_model(struct LinearRegression *linreg, int num_features)
{
    linreg->w = calloc(num_features, sizeof(double));
    if (!linreg->w)
    {
        printf("Unable to allocate memory for weights.\n");
        exit(EXIT_FAILURE);
    }
    linreg->b = 0.0;
}

// Fit linear regression model to training data
void fit_model(struct LinearRegression *linreg, struct Dataset *data)
{
    // Retrieve dataset counts and model parameters
    int num_features = data->num_features;
    int train_length = data->train_length;

    int num_epochs = linreg->num_epochs;
    double learning_rate = linreg->learning_rate;
    int batch_size = linreg->batch_size;
    double l2_alpha = linreg->l2_alpha;
    double l1_alpha = linreg->l1_alpha;
    double r = linreg->mix_ratio;

    double w[num_features];
    memcpy(w, linreg->w, num_features * sizeof(double));
    double b = linreg->b;

    // If necessary, implement test MSE evaluation for early stopping
    int early_stopping = linreg->early_stopping;
    double valid_MSE = 0.0;
    double prev_valid_MSE = -1.0;
    double sensitivity = 0.005; // Minimum acceptable decrease in MSE
    int patience = 2; // Number of "bad" MSE comparisons required in a row
    int patience_counter = 0;

    // Gradient accumulation array
    double *w_sums = calloc(num_features, sizeof(double));
    if (!w_sums)
    {
        printf("Unable to allocate memory for w_sums.\n");
        exit(EXIT_FAILURE);
    }

    // Validate chosen batch size
    if (batch_size > train_length) {
        printf("Batch size cannot be greater than size of training set, Batch Size: %i, Training Size: %i\n.", batch_size, train_length);
        exit(EXIT_FAILURE);
    }

    // Adjust training length to batch size
    if (batch_size != 0)
    {
        train_length = batch_size;
    }

    // Print terminal log header
    printf("==========================================\n");
    printf("| Epoch |   Train MSE   | Validation MSE |\n");
    printf("==========================================\n");

    // Iterate epochs
    for (int epoch = 0; epoch <= num_epochs; epoch++)
    {
        // Generate smaller batch, if necessary
        if (batch_size != 0)
        {
            // "Generate" a random batch by swapping training samples
            shuffle_batch(data, batch_size);
        }

        // Reset gradient accumulators
        memset(w_sums, 0, num_features * sizeof(double));
        double b_sum = 0.0;

        // Iterate training data, accumulating gradients
        for (int i = 0; i < train_length; i++) // "i" loops iterate training samples
        {
            // Make a prediction, y-hat
            double y_pred = b;
            // Sum weighted feature contributions
            for (int j = 0; j < num_features; j++)
            {
                y_pred += w[j] * data->X_train[i * num_features + j];
            }

            // Calculate error
            double error = data->y_train[i] - y_pred;

            // Accumulate gradients
            b_sum += error;
            // For each feature
            for (int j= 0; j < num_features;j++)
            {
                w_sums[j] += data->X_train[i * num_features + j] * error;
            }
        }

        // Calculate partial derivative of MSE with respect to w
        for (int j = 0; j < num_features; j++)
        {
            // Base derivative of objective function
            double dE_dw = (-2.0 / train_length) * w_sums[j];

            // Apply regularization gradient
            dE_dw += gradient_regularization(w[j], train_length, l2_alpha, l1_alpha, r);

            // Update feature weight, compensating for learning rate
            w[j] -= learning_rate * dE_dw;
        }

        // Calculate partial derivative of MSE with respect to b
        double dE_db = (-2.0 / train_length) * b_sum;
        // Update bias parameter, compensating for learning rate
        b -= learning_rate * dE_db;

        // Print training progress intermittently
        int divisor = (num_epochs / 10 == 0) ? 1 : num_epochs / 10; // Prevent dividing by 0 with small epoch number
        if (epoch % divisor == 0)
        {
            // Copy weights and bias back to model
            for (int j = 0; j < num_features; j++)
            {
                linreg->w[j] = w[j];
            }
            linreg->b = b;

            // Make predictions and calculate training MSE
            predict(linreg, data, data->X_train, data->train_length);
            double MSE = mean_squared_error(data->y_train, data->y_pred, data->train_length);
            printf("| %5d | %13.5f |", epoch, MSE);

            // Evaluate for validation set/early stopping
            if (data->valid_proportion == 0.0 || !early_stopping)
            {
                printf("       N/A      |\n");
                continue;
            }

            // Make predictions and calculate validation MSE
            predict(linreg, data, data->X_valid, data->valid_length);
            valid_MSE = mean_squared_error(data->y_valid, data->y_pred, data->valid_length);
            printf(" %14.5f |\n", valid_MSE);

            // If applicable, set as initial error and continue
            if (prev_valid_MSE == -1.0)
            {
                prev_valid_MSE = valid_MSE;
                continue;
            }

            // Evaluate MSE difference and current patience counter to consider early stopping
            if (valid_MSE >= prev_valid_MSE || (prev_valid_MSE - valid_MSE) < sensitivity)
            {
                patience_counter++;
                if (patience_counter >= patience) {
                    // Print footer of terminal log and exit early
                    printf("------------------------------------------\n");
                    printf("Stopping early, validation set has reached a minimum error.\n");
                    free(w_sums);
                    return;
                }
            }
            else
            {
                patience_counter = 0;
            }
            prev_valid_MSE = valid_MSE;
        }
    }

    // Final weights and bias update
    for (int j = 0; j < num_features; j++)
    {
        linreg->w[j] = w[j];
    }
    linreg->b = b;

    // Print footer of terminal log
    printf("------------------------------------------\n");

    // Free gradient accumulator array
    free(w_sums);
}

// Print trained weight and bias parameters, un-standardized if necessary
void print_model_parameters(struct LinearRegression *linreg, struct Dataset *data)
{
    int num_features = data->num_features;
    int polynomial_degree = linreg->polynomial_degree;
    int og_num_features = num_features / polynomial_degree;

    printf("\nWeights:\n");
    for (int j = 0; j < og_num_features; j++)
    {
        printf("    Feature %3d: ",j + 1);
        for (int d = 1; d <= polynomial_degree; d++)
        {
            int weight_index = (j * polynomial_degree) + (d - 1);
            // Un-standardize model weight, if necessary
            double weight = linreg->w[weight_index];
            if (data->standardized)
            {
                weight /= data->feature_stds[j];
            }

            // Print weight
            printf(" x^%d: %6.3f", d, weight);
            if (d != polynomial_degree)
            {
                printf(", ");
            }
            else
            {
                printf("\n");
            }
        }

    }

    // Un-standardize bias, if necessary
    double bias = linreg->b;
    if (data->standardized)
    {
        for (int j = 0; j < og_num_features; j++)
        {
            // Parse original data mean and standard deviation
            double feature_mean = data->feature_means[j * polynomial_degree];
            double feature_std = data->feature_stds[j * polynomial_degree];
            for (int d = 1; d <= polynomial_degree; d++)
            {
                int weight_index = (j * polynomial_degree) + (d - 1);
                bias -= (linreg->w[weight_index] * feature_mean) / feature_std;
            }
        }
    }
    printf("Bias: %.3f\n", bias);
}

// Make predictions using trained linear regression model
void predict(struct LinearRegression *linreg, struct Dataset *data, double *X_predict, int num_predictions)
{
    int num_features = data->num_features;

    // Reallocate y_pred array for size of predictions
    data->y_pred = realloc(data->y_pred, num_predictions * sizeof(double));
    if (!data->y_pred)
    {
        printf("Unable to reallocate y_pred.\n");
        exit(EXIT_FAILURE);
    }

    double b = linreg->b;

    // Calculate and store predictions
    for (int i = 0; i < num_predictions; i++)
    {
        double temp_prediction = b;
        // Sum weighted feature contributions
        for (int j = 0; j < num_features; j++)
        {
            temp_prediction += linreg->w[j] * X_predict[i * num_features + j];
        }
        data->y_pred[i] = temp_prediction;
    }
}
