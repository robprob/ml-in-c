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

#include "mlper.h"

// Linear Regression model with default hyperparameters
struct LinearRegression {
    int num_epochs;            // Number of training iterations
    double learning_rate;      // Training "step" size
    double test_proportion;    // Proportion of training data held in test set
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
void fit_model(struct LinearRegression *linreg, struct Dataset *data);
void predict(struct LinearRegression *linreg, struct Dataset *data, double *X_predict, int num_predictions);


int main(int argc, char **argv)
{
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
    printf("Test Proportion: %g\n", linreg.test_proportion);
    printf("Early Stopping: %i\n", linreg.early_stopping);
    printf("Batch Size: %i\n", linreg.batch_size);
    printf("L2 alpha: %g\n", linreg.l2_alpha);
    printf("L1 alpha: %g\n", linreg.l1_alpha);
    printf("Elastic Net Mix Ratio: %g\n", linreg.mix_ratio);
    */

    // Load feature and target variable data into arrays
    load(&data);

    // Split data into training and test arrays
    train_test_split(&data, linreg.test_proportion);

    // If specified, standardize feature arrays to mean of 0, standard deviation of 1
    if (data.standardized)
    {
        standardize(&data);
    }

    // Initialize weight vector and bias at 0 (horizontal line)
    linreg.w = calloc(data.num_features, sizeof(double));
    if (!linreg.w)
    {
        printf("Unable to allocate memory for weights.\n");
    }
    linreg.b = 0;

    clock_t start, end;

    start = clock();

    // Fit model to training data
    fit_model(&linreg, &data);

    end = clock();

    // Print trained model parameters
    printf("\nTrained Model Parameters (un-standardized)\n");
    printf("Weights: ");
    for (int j = 0; j < data.num_features; j++)
    {
        // Un-standardize model weights
        double weight = linreg.w[j] / data.feature_stds[j];
        printf("%g", weight);
        if (j != data.num_features - 1)
        {
            printf(", ");
        }
        else
        {
            printf("\n");
        }
    }
    // Un-standardize bias
    double bias = linreg.b;
    for (int j = 0; j < data.num_features; j++)
    {
        bias -= (linreg.w[j] * data.feature_means[j]) / data.feature_stds[j];
    }
    printf("Bias: %g\n", bias);

    // Generate predictions using trained model
    predict(&linreg, &data, data.X_test, data.test_length);

    // Calculate Mean Squared Error
    double mse = mean_squared_error(data.y_test, data.y_pred, data.test_length);
    printf("\nTest MSE: %f\n", mse);

    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Training Time: %f seconds\n", cpu_time);

    // Unstandardize input matrix, if necessary
    if (data.standardized)
    {
        unstandardize(&data, data.X_test, data.test_length);
    }

    // Export feature data and calculated predictions
    export_results(&data, data.test_length, "test_predictions.csv");

    // Free memory taken up by dataset
    free_dataset(&data);
    // Free parameter weights array
    free(linreg.w);
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
    char line[128];
    int line_num = 1;

    while (fgets(line, sizeof(line), file))
    {
        char value[128];
        sscanf(line, "%*[^=]=%s", value);

        switch (line_num++)
        {
            case 1:
                strcpy(data->file_path, value);
                break;
            case 2:
                if (strcmp(value, "true") == 0)
                {
                    data->standardized = 1;
                }
                else if (strcmp(value, "false") == 0)
                {
                    data->standardized = 0;
                }
                else
                {
                    printf("Standardized options: 'true' or 'false'\n");
                }
                break;
            case 3:
                linreg->num_epochs = atoi(value);
                break;
            case 4:
                linreg->learning_rate = atof(value);
                break;
            case 5:
                linreg->test_proportion = atof(value);
                break;
            case 6:
                if (strcmp(value, "true") == 0)
                {
                    linreg->early_stopping = 1;
                }
                else if (strcmp(value, "false") == 0)
                {
                    linreg->early_stopping = 0;
                }
                else
                {
                    printf("Early Stopping options: 'true' or 'false'\n");
                }
                break;
            case 7:
                linreg->batch_size = atoi(value);
                break;
            case 8:
                linreg->l2_alpha = atof(value);
                break;
            case 9:
                linreg->l1_alpha = atof(value);
                break;
            case 10:
                linreg->mix_ratio = atof(value);
                break;
            case 11:
                printf("Too many configuration keys\n");
                exit(EXIT_FAILURE);
        }
    }
    fclose(file);
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
    double test_MSE = 0.0;
    double prev_test_MSE = 0.0;
    double sensitivity = 0.0001; // Minimum acceptable decrease in MSE

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

    // Iterate epochs
    for (int epoch = 0; epoch <= num_epochs; epoch++)
    {
        // Generate smaller batch, if necessary
        if (batch_size != 0)
        {
            // "Generate" a random batch by swapping training entries
            shuffle_batch(data, batch_size);
        }

        // Reset gradient accumulators
        memset(w_sums, 0, num_features * sizeof(double));
        double b_sum = 0;

        // Iterate training data, accumulating gradients
        for (int i = 0; i < train_length; i++) // "i" loops iterate training entries
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

            // Check for Elastic Net, adjusting coefficient accordingly
            if (l2_alpha > 0.0 && l1_alpha > 0.0)
            {
                l2_alpha = (1 - r) * l2_alpha;
                l1_alpha = r * l1_alpha;
            }

            // Add L2 regularization (ridge)
            if (l2_alpha > 0.0)
            {
                dE_dw += (2.0 / train_length) * l2_alpha * w[j];
            }

            // Add L1 regularization (lasso)
            if (l1_alpha > 0.0)
            {
                // Derivative of absolute value is dependent on sign of weight
                if (w[j] > 0)
                {
                    dE_dw += l1_alpha;
                }
                else if (w[j] < 0)
                {
                    dE_dw -= l1_alpha;
                }
            }
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

            // Make predictions and calculate MSE
            predict(linreg, data, data->X_train, data->train_length);
            double MSE = mean_squared_error(data->y_train, data->y_pred, data->train_length);
            printf("Epoch %d: Train MSE: %f\n", epoch, MSE);


            // Evaluate test MSE
            predict(linreg, data, data->X_test, data->test_length);
            test_MSE = mean_squared_error(data->y_test, data->y_pred, data->test_length);

            // Evaluate for selection of early stopping
            if (!early_stopping)
            {
                continue;
            }

            // If applicable, set as initial error and continue
            if (prev_test_MSE == 0.0)
            {
                prev_test_MSE = MSE;
                continue;
            }

            // Compare difference to consider early stopping
            double diff = test_MSE - prev_test_MSE;
            if (diff > 0.0 || diff > (sensitivity * -1))
            {
                printf("Stopping early, validation set has reached a minimum error.\n");
                free(w_sums);
                return;
            }
            prev_test_MSE = test_MSE;
        }
    }

    // Final weights and bias update
    for (int j = 0; j < num_features; j++)
    {
        linreg->w[j] = w[j];
    }
    linreg->b = b;

    // Free gradient accumulator array
    free(w_sums);
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
