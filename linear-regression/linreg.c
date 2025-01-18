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
    int num_epochs; // Number of training iterations
    double learning_rate; // Training "step" size
    double test_proportion; // Proportion of training data held in test set
    char gradient_descent[10]; // Variant of GD
    int batch_size; // Number of entries selected per epoch
    double l2_lambda; // Ridge
    double l1_lambda; // Lasso
    double *w; // Weight vector
    double b; // Bias
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

    // Print selected parameters
    printf("File Path: %s\n", data.file_path);
    printf("Number of Epochs: %i\n", linreg.num_epochs);
    printf("Learning Rate: %g\n", linreg.learning_rate);
    printf("Test Proportion: %g\n", linreg.test_proportion);
    printf("Batch Size: %i\n", linreg.batch_size);
    printf("Gradient Descent: %s\n", linreg.gradient_descent);
    printf("L2 Lambda: %g\n", linreg.l2_lambda);
    printf("L1 Lambda: %g\n", linreg.l1_lambda);

    // Load feature and target variable data into arrays
    load(&data);

    // Standardize feature data, X, to mean of 0, standard deviation of 1
    standardize(&data);

    // Split data into training and test arrays
    train_test_split(&data, linreg.test_proportion);

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

    // Generate predictions using trained model
    predict(&linreg, &data, data.X_test, data.test_length);

    // Calculate Mean Squared Error
    double mse = mean_squared_error(data.y_test, data.y_pred, data.test_length);
    printf("\nTest MSE: %f\n", mse);

    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Training Time : %f seconds\n", cpu_time);

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
                linreg->num_epochs = atoi(value);
                break;
            case 3:
                linreg->learning_rate = atof(value);
                break;
            case 4:
                linreg->test_proportion = atof(value);
                break;
            case 5:
                strcpy(linreg->gradient_descent, value);
                break;
            case 6:
                linreg->batch_size = atoi(value);
                break;
            case 7:
                linreg->l2_lambda = atof(value);
                break;
            case 8:
                linreg->l1_lambda = atof(value);
                break;
            case 9:
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
    double l2_lambda = linreg->l2_lambda;
    double l1_lambda = linreg->l1_lambda;

    double w[num_features];
    memcpy(w, linreg->w, num_features * sizeof(double));
    double b = linreg->b;

    // Gradient accumulation array
    double *w_sums = calloc(num_features, sizeof(double));
    if (!w_sums)
    {
        printf("Unable to allocate memory for w_sums.\n");
        exit(EXIT_FAILURE);
    }

    // Iterate epochs
    for (int epoch = 0; epoch <= num_epochs; epoch++)
    {
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

            // Add L2 regularization (ridge)
            if (l2_lambda > 0.0)
            {
                dE_dw += (2.0 / train_length) * l2_lambda * w[j];
            }

            // Add L1 regularization (lasso)
            if (l1_lambda > 0.0)
            {
                // Derivative of absolute value is dependent on sign of weight
                if (w[j] > 0)
                {
                    dE_dw += l1_lambda;
                }
                else if (w[j] < 0)
                {
                    dE_dw -= l1_lambda;
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
