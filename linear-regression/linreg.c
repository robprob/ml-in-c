/*
Implementation of multi-variable linear regression, including L2/L1 regularization.

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

// Linear Regression model with parameters
struct LinearRegression {
    double *w; // Weight vector
    double b; // Bias
    double l2_lambda; // Ridge
    double l1_lambda; // Lasso
};

// Function Prototypes
void print_usage();
struct LinearRegression fit_model(struct LinearRegression linreg, int num_epochs, double learning_rate);
void predict(struct LinearRegression linreg, double *X_predict, int num_predictions);


int main(int argc, char **argv)
{
    // Instantiate linear regression model, initialized to 0
    struct LinearRegression linreg = {0};

    // Validate command-line arguments
    char *file_path;
    int num_epochs;
    double learning_rate;

    // Parse data file, number of epochs, and learning rate
    if (argc <= 2)
    {
        if (argc == 1)
        {
            file_path = "../sample-data/linear_multi_var_500.csv";
        }
        else if (strcmp(argv[1], "--help") == 0)
        {
            print_usage();
            return 0;
        }
        else
        {
            file_path = argv[1];
        }

        num_epochs = 100;
        learning_rate = 0.05;

        linreg.l2_lambda = 0.0;
        linreg.l1_lambda = 0.0;
    }
    else if (argc == 4 || argc == 6 || argc == 8)
    {
        file_path = argv[1];
        num_epochs = atoi(argv[2]);
        learning_rate = atof(argv[3]);
    }
    else
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    // Parse regularization inputs
    if (argc == 4)
    {
        linreg.l2_lambda = 0.0;
        linreg.l1_lambda = 0.0;
    }
    else if (argc == 6)
    {
        if (strcmp(argv[4], "--ridge") == 0)
        {
            linreg.l2_lambda = atof(argv[5]);
        }
        else if (strcmp(argv[4], "--lasso") == 0)
        {
            linreg.l1_lambda = atof(argv[5]);
        }
        else
        {
            print_usage();
            exit(EXIT_FAILURE);
        }
    }
    else if (argc == 8)
    {
        linreg.l2_lambda = atof(argv[5]);
        linreg.l1_lambda = atof(argv[7]);
    }

    // Final validation of numeric inputs
    if (num_epochs <= 0 || learning_rate <= 0)
    {
        printf("num_epochs: %i, and learning_rate: %f must be positive numbers.\n", num_epochs, learning_rate);
        exit(EXIT_FAILURE);
    }

    // Print selected parameters
    printf("File Path: %s\n", file_path);
    printf("Number of Epochs: %i\n", num_epochs);
    printf("Learning Rate: %g\n", learning_rate);
    printf("L2 Lambda: %g\n", linreg.l2_lambda);
    printf("L1 Lambda: %g\n", linreg.l1_lambda);

    // Load feature and target variable data into arrays
    load(file_path);

    // Standardize feature data, X, to mean of 0, standard deviation of 1
    standardize(X, num_features, num_entries);

    // Split data into training and test arrays
    train_test_split(0.3);

    // Initialize weight vector and bias at 0 (horizontal line)
    linreg.w = calloc(num_features, sizeof(double));
    if (!linreg.w)
    {
        printf("Unable to allocate memory for weights.\n");
    }
    linreg.b = 0;

    clock_t start, end;

    start = clock();

    // Fit model to training data with specified epochs and training rate
    linreg = fit_model(linreg, num_epochs, learning_rate);

    end = clock();

    // Generate predictions using trained model
    predict(linreg, X_test, test_length);

    // Calculate Mean Squared Error
    double mse = mean_squared_error(y_test, y_pred, test_length);
    printf("\nTest MSE: %f\n", mse);

    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Training Time : %f seconds\n", cpu_time);

    // Export feature data and calculated predictions
    export_results("test_predictions.csv", X_test, y_test, y_pred, num_features, test_length);

    // Free memory taken up by dataset
    free_globals();
    // Free parameter weights array
    free(linreg.w);
}

// Display command-line usage information
void print_usage()
{
    printf("Usage:\n");
    printf("  ./linreg\n");
    printf("  ./linreg [data.csv]\n");
    printf("  ./linreg [data.csv] [num_epochs] [learning_rate] [options]\n");
    printf("Options:\n");
    printf("  --ridge [l2_lambda]      Use Ridge regression with L2 regularization\n");
    printf("  --lasso [l1_lambda]      Use Lasso regression with L1 regularization\n");
    printf("  --help                   Display this help message\n");
}

// Fit linear regression model to training data
struct LinearRegression fit_model(struct LinearRegression linreg, int num_epochs, double learning_rate)
{
    // Create new temporary array of weights
    double w[num_features];
    memcpy(w, linreg.w, num_features * sizeof(double));

    // Bias parameter
    double b = linreg.b;

    // Gradient accumulation
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
                y_pred += w[j] * X_train[i * num_features + j];
            }

            // Calculate error
            double error = y_train[i] - y_pred;

            // Accumulate gradients
            b_sum += error;
            // For each feature
            for (int j= 0; j < num_features;j++)
            {
                w_sums[j] += X_train[i * num_features + j] * error;
            }
        }

        // Calculate partial derivative of MSE with respect to w
        for (int j = 0; j < num_features; j++)
        {
            // Base derivative of objective function
            double dE_dw = (-2.0 / train_length) * w_sums[j];

            // Add L2 regularization (ridge)
            if (linreg.l2_lambda > 0.0)
            {
                dE_dw += 2 * linreg.l2_lambda * w[j];
            }

            // Add L1 regularization (lasso)
            if (linreg.l1_lambda > 0.0)
            {
                // Derivative of absolute value is dependent on sign of weight
                if (w[j] > 0)
                {
                    dE_dw += linreg.l1_lambda;
                }
                else if (w[j] < 0)
                {
                    dE_dw -= linreg.l1_lambda;
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
        int divisor = (num_epochs / 10 == 0) ? 1 : num_epochs / 10;
        if (epoch % divisor == 0)
        {
            // Copy weights and bias back to model
            for (int j = 0; j < num_features; j++)
            {
                linreg.w[j] = w[j];
            }
            linreg.b = b;

            // Make predictions and calculate MSE
            predict(linreg, X_train, train_length);
            double MSE = mean_squared_error(y_train, y_pred, train_length);
            printf("Epoch %d: Train MSE: %f\n", epoch, MSE);
        }
    }

    // Final weights and bias update
    for (int j = 0; j < num_features; j++)
    {
        linreg.w[j] = w[j];
    }
    linreg.b = b;

    // Free gradient accumulator array
    free(w_sums);
    return linreg;
}

// Make predictions using trained linear regression model
void predict(struct LinearRegression linreg, double *X_predict, int num_predictions)
{
    // Reallocate global y_pred array for size of predictions
    y_pred = realloc(y_pred, num_predictions * sizeof(double));
    if (!y_pred)
    {
        printf("Unable to reallocate y_pred.\n");
        exit(EXIT_FAILURE);
    }

    // Retrieve model parameters
    double b = linreg.b;

    // Calculate and store predictions
    for (int i = 0; i < num_predictions; i++)
    {
        double temp_prediction = b;
        // Sum weighted feature contributions
        for (int j = 0; j < num_features; j++)
        {
            temp_prediction += linreg.w[j] * X_predict[i * num_features + j];
        }
        y_pred[i] = temp_prediction;
    }
}
