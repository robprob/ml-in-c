/*

Implementation of various Generalized Linear Models

Gaussian
Poisson
Gamma
Tweedie

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "mlper.h"

// Define function pointers for link and loss
typedef double (*InverseLinkFunction)(double);
typedef double (*ResidualFunction)(double, double);
typedef double (*LossFunction)(double, double);

struct GeneralizedRegressor {
    char distribution_type[20];         // Generalized linear model distribution
    double p;                           // Power paramater for Tweedie distribution
    int num_epochs;                     // Number of training iterations
    int polynomial_degree;              // Highest polynomial degree to generate from features
    double learning_rate;               // Training "step" size
    int early_stopping;                 // Truth value of early stopping
    double tolerance;                   // Minimum acceptable reduction in loss
    int patience;                       // Number of "bad" loss checks required in a row
    int batch_size;                     // Number of samples taken per epoch (leave at 0 for batch GD)
    double l2_alpha;                    // Ridge coefficient
    double l1_alpha;                    // Lasso coefficient
    double mix_ratio;                   // 0 is equivalent to pure ridge, 1 is equivalent to pure lasso
    double *w;                          // Weight vector
    double b;                           // Bias
    InverseLinkFunction inverse_link;   // Inverse of link function for specified regression type
    ResidualFunction residual_function; // Derivative of inverse link used for calculating residual
    LossFunction loss_function;         // Loss function used to evaluate model fit
};

// Function Prototypes
void parse_config(struct Dataset *data, struct GeneralizedRegressor *genreg);
void initialize_model(struct GeneralizedRegressor *genreg, int num_features);
void fit_model(struct GeneralizedRegressor *genreg, struct Dataset *data);
void print_model_parameters(struct GeneralizedRegressor *genreg, struct Dataset *data);
void predict(struct GeneralizedRegressor *genreg, struct Dataset *data, double *X_predict, int num_predictions);
void predict_class(double *y_pred, int num_predictions);
double average_loss(struct GeneralizedRegressor *genreg, double *y_true, double *y_pred, int num_predictions);


int main(int argc, char **argv)
{
    clock_t start, end;

    start = clock();

    // Instantiate dataset at 0
    struct Dataset data = {0};
    // Instantiate Linear Regression model at 0
    struct GeneralizedRegressor genreg = {0};

    // Parse data path and model hyperparameter from config file
    parse_config(&data, &genreg);

    #ifdef VERBOSE
        // Print parsed parameters
        printf("File Path: %s\n", data.file_path);
        printf("Standardized: %i\n", data.standardized);
        printf("Distribution Type: %s\n", genreg.distribution_type);
        printf("Power Paramater: %g\n", genreg.p);
        printf("Max Polynomial Degree: %i\n", genreg.polynomial_degree);
        printf("Number of Epochs: %i\n", genreg.num_epochs);
        printf("Learning Rate: %g\n", genreg.learning_rate);
        printf("Test Proportion: %g\n", data.test_proportion);
        printf("Validation Proportion: %g\n", data.valid_proportion);
        printf("Early Stopping: %i\n", genreg.early_stopping);
        printf("Batch Size: %i\n", genreg.batch_size);
        printf("L2 alpha: %g\n", genreg.l2_alpha);
        printf("L1 alpha: %g\n", genreg.l1_alpha);
        printf("Elastic Net Mix Ratio: %g\n", genreg.mix_ratio);
        printf("Tolerance: %g\n", genreg.tolerance);
        printf("Patience %i\n", genreg.patience);
    #endif

    // Load feature and target variable data into arrays
    load(&data);
    // Transform feature matrix to specified highest degree
    poly_transform(&data, genreg.polynomial_degree);
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

    // Initialize starting model parameters
    initialize_model(&genreg, data.num_features);

    clock_t train_start, train_end;

    train_start = clock();

    // Fit model to training data
    fit_model(&genreg, &data);

    train_end = clock();

    // Print trained model parameters, un-standardized if necessary
    print_model_parameters(&genreg, &data);

    // Generate predictions using trained model
    predict(&genreg, &data, data.X_test, data.test_length);

    // Calculate test set loss
    printf("\nTest Loss: %g\n", average_loss(&genreg, data.y_test, data.y_pred, data.test_length));

    // Print score/performance based on chosen model
    print_model_performance(genreg.distribution_type, data.y_test, data.y_pred, data.test_length);

    // If necessary, convert existing probability predictions into discrete class predictions
    if (strcmp(genreg.distribution_type, "poisson") == 0)
    {
        predict_class(data.y_pred, data.test_length);
    }

    // Calculate and print model training time
    double training_time = ((double) (train_end - train_start)) / CLOCKS_PER_SEC;
    printf("\nTraining Time: %f seconds\n", training_time);

    // Export feature data and calculated predictions
    export_results(&data, data.test_length, genreg.polynomial_degree, "test_predictions.csv");

    // Free memory taken up by dataset
    free_dataset(&data);
    // Free parameter weights array
    free(genreg.w);

    end = clock();
    double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Total CPU Time: %f seconds\n", total_time);
}

// Parse data file path and model hyperparameter from config file
void parse_config(struct Dataset *data, struct GeneralizedRegressor *genreg)
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
            if (strcmp(key, "distribution_type ") == 0)
            {
                strcpy(genreg->distribution_type, value);
            }
            else if (strcmp(key, "p ") == 0)
            {
                if (strcmp(genreg->distribution_type, "tweedie") == 0)
                {
                    genreg->p = atof(value);
                }
                else
                {
                    genreg->p = -1;
                }
            }
            else if (strcmp(key, "file_path ") == 0)
            {
                strcpy(data->file_path, value);
            }
            else if (strcmp(key, "polynomial_degree ") == 0)
            {
                genreg->polynomial_degree = atoi(value);
            }
            else if (strcmp(key, "standardize ") == 0)
            {
                data->standardized = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "num_epochs ") == 0)
            {
                genreg->num_epochs = atoi(value);
            }
            else if (strcmp(key, "learning_rate ") == 0)
            {
                genreg->learning_rate = atof(value);
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
                genreg->early_stopping = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "batch_size ") == 0)
            {
                genreg->batch_size = atoi(value);
            }
            else if (strcmp(key, "l2_alpha ") == 0)
            {
                genreg->l2_alpha = atof(value);
            }
            else if (strcmp(key, "l1_alpha ") == 0)
            {
                genreg->l1_alpha = atof(value);
            }
            else if (strcmp(key, "mix_ratio ") == 0)
            {
                genreg->mix_ratio = atof(value);
            }
            else if (strcmp(key, "tolerance ") == 0)
            {
                genreg->tolerance = atof(value);
            }
            else if (strcmp(key, "patience ") == 0)
            {
                genreg->patience = atoi(value);
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

// Initialize parameters and functions for chosen model
void initialize_model(struct GeneralizedRegressor *genreg, int num_features)
{
    // Initialize memory for weights at 0
    genreg->w = calloc(num_features, sizeof(double));
    if (!genreg->w)
    {
        printf("Unable to allocate memory for weights.\n");
        exit(EXIT_FAILURE);
    }
    // Randomly initialize weights via Xavier/Glorot, so weights don't get "stuck" near 0
    xavier_glorot_init(genreg->w, num_features);

    // Initialize bias
    genreg->b = 0.0;

    // Set link/inverse link functions
    if (strcmp(genreg->distribution_type, "gaussian") == 0 || strcmp(genreg->distribution_type, "normal") == 0)
    {
        // No transformation done, linear link is reflexive
        genreg->inverse_link = identity_link;
        genreg->residual_function = gaussian_residual;
        genreg->loss_function = gaussian_nll;
    }
    else if (strcmp(genreg->distribution_type, "poisson") == 0)
    {
        genreg->inverse_link = inverse_log_link;
        genreg->residual_function = poisson_residual;
        genreg->loss_function = poisson_nll;
    }
    else if (strcmp(genreg->distribution_type, "gamma") == 0)
    {
        genreg->inverse_link = inverse_log_link;
        genreg->residual_function = gamma_residual;
        genreg->loss_function = gamma_loss;
    }
    else if (strcmp(genreg->distribution_type, "tweedie") == 0)
    {
        if (genreg->p == 0)
        {
            genreg->inverse_link = identity_link;
        }
        else if (genreg->p == 1 || genreg->p == 2)
        {
            genreg->inverse_link = inverse_log_link;
        }
    }
    else
    {
        printf("Invalid distribution. Exiting.\n");
        exit(EXIT_FAILURE);
    }
}

// Fit linear regression model to training data
void fit_model(struct GeneralizedRegressor *genreg, struct Dataset *data)
{
    // Retrieve dataset counts and model parameters
    int num_features = data->num_features;
    int train_length = data->train_length;

    int num_epochs = genreg->num_epochs;
    double learning_rate = genreg->learning_rate;
    int batch_size = genreg->batch_size;

    // Distribution paramaters
    double p = genreg->p;

    // Regularization hyperparameters
    double l2_alpha = genreg->l2_alpha;
    double l1_alpha = genreg->l1_alpha;
    double r = genreg->mix_ratio;

    double w[num_features];
    memcpy(w, genreg->w, num_features * sizeof(double));
    double b = genreg->b;

    // Early stopping variables
    int early_stopping = genreg->early_stopping;
    double valid_loss = 0.0;
    double prev_valid_loss = -1.0;
    double tolerance = genreg->tolerance;
    int patience = genreg->patience;
    int patience_counter = 0;

    // Gradient accumulation array
    double *w_grad = calloc(num_features, sizeof(double));
    if (!w_grad)
    {
        printf("Unable to allocate memory for w_grad.\n");
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
    printf("| Epoch |  Train Loss  | Validation Loss |\n");
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

        // Initialize/reset gradient accumulators
        // Reset gradient accumulators
        memset(w_grad, 0, num_features * sizeof(double));
        double b_grad = 0.0;

        // Calculate total loss with existing weight and bias parameters
        double total_loss = 0.0;
        double reg_loss = 0.0; // Regularization loss penalty

        // Iterate training data, accumulating gradients
        for (int i = 0; i < train_length; i++) // "i" loops iterate training samples
        {
            // Calculate the linear combination, eta
            double eta = dot_prod(data->X_train, i, w, num_features) + b;

            // Predict mean using inverse link function
            double y_pred = (p == -1 || p == 1 || p == 2) ? genreg->inverse_link(eta) : inverse_tweedie_link(eta, p);

            // Compute residual
            double y_true = data->y_train[i];
            double residual = (p == -1) ? genreg->residual_function(y_true, y_pred) : tweedie_residual(y_true, y_pred, p);

            // Accumulate loss
            total_loss += (p == -1) ? genreg->loss_function(y_true, y_pred) : tweedie_nll(y_true, y_pred, p);

            // Accumulate gradients
            for (int j = 0; j < num_features; j++)
            {
                w_grad[j] += data->X_train[i * num_features + j] * residual;
            }
            b_grad += residual;
        }

        // Add any regularization to gradient and loss
        for (int j = 0; j < num_features; j++)
        {
            w_grad[j] += regularization_gradient(w[j], l2_alpha, l1_alpha, r);
            reg_loss += regularization_loss(w[j], l2_alpha, l1_alpha, r);
        }

        // Update weights and bias parameters using accumulated gradients
        for (int j = 0; j < num_features; j++)
        {
            w[j] -= learning_rate * w_grad[j] / train_length;
        }
        b -= learning_rate * b_grad / train_length;

        // Compute average loss
        double avg_loss = (total_loss / train_length) + reg_loss;

        // Print training progress intermittently
        int divisor = (num_epochs / 10 == 0) ? 1 : num_epochs / 10; // Prevent dividing by 0 with small epoch number
        if (epoch % divisor == 0)
        {
            // Check for validation set
            if (data->valid_proportion == 0.0)
            {
                printf("| %5d | %12.5f |", epoch, avg_loss);
                printf("       N/A       |\n");
                continue;
            }

            // Copy weights and bias parameters back to model
            memcpy(genreg->w, w, num_features * sizeof(double));
            genreg->b = b;

            // Calculate probabilities and calculate average loss of validation set
            predict(genreg, data, data->X_valid, data->valid_length);
            valid_loss = average_loss(genreg, data->y_valid, data->y_pred, data->valid_length);

            printf("| %5d | %12.5f | %15.5f |\n", epoch, avg_loss, valid_loss);

            // Check for early stopping
            if (!early_stopping)
            {
                continue;
            }

            // If applicable, set as initial error and continue
            if (prev_valid_loss == -1.0)
            {
                prev_valid_loss = valid_loss;
                continue;
            }

            // Evaluate MSE difference and current patience counter to consider early stopping
            if (valid_loss >= prev_valid_loss || (prev_valid_loss - valid_loss) < tolerance)
            {
                patience_counter++;
                if (patience_counter >= patience) {
                    // Print footer of terminal log and exit early
                    printf("------------------------------------------\n");
                    printf("Stopping early at epoch %d, validation set has reached a minimum loss.\n", epoch);
                    free(w_grad);
                    return;
                }
            }
            else
            {
                patience_counter = 0;
            }
            prev_valid_loss = valid_loss;
        }
    }

    // Final weights and bias update
    memcpy(genreg->w, w, num_features * sizeof(double));
    genreg->b = b;

    // Print footer of terminal log
    printf("------------------------------------------\n");

    // Free gradient accumulator
    free(w_grad);
}

// Print trained weight and bias parameters, un-standardized if necessary
void print_model_parameters(struct GeneralizedRegressor *genreg, struct Dataset *data)
{
    int num_features = data->num_features;
    int polynomial_degree = genreg->polynomial_degree;
    int og_num_features = num_features / polynomial_degree;

    printf("\nWeights:\n");
    for (int j = 0; j < og_num_features; j++)
    {
        printf("    Feature %3d: ",j + 1);
        for (int d = 1; d <= polynomial_degree; d++)
        {
            int weight_index = (j * polynomial_degree) + (d - 1);
            // Un-standardize model weight, if necessary
            double weight = genreg->w[weight_index];
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
    double bias = genreg->b;
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
                bias -= (genreg->w[weight_index] * feature_mean) / feature_std;
            }
        }
    }
    printf("Bias: %.3f\n", bias);
}

// Make predictions using trained regression model
void predict(struct GeneralizedRegressor *genreg, struct Dataset *data, double *X_predict, int num_predictions)
{
    int num_features = data->num_features;
    double p = genreg->p;

    // Reallocate y_pred array for size of predictions
    data->y_pred = realloc(data->y_pred, num_predictions * sizeof(double));
    if (!data->y_pred)
    {
        printf("Unable to reallocate y_pred.\n");
        exit(EXIT_FAILURE);
    }

    double b = genreg->b;

    // Calculate and store predictions
    for (int i = 0; i < num_predictions; i++)
    {
        // Calculate the linear combination, eta
        double eta = dot_prod(X_predict, i, genreg->w, num_features) + b;

        // Predict mean using inverse link function
        data->y_pred[i] = (p == -1 || p == 1 || p == 2) ? genreg->inverse_link(eta) : inverse_tweedie_link(eta, p);
    }
}

// Convert existing probability predictions into discrete class predictions
void predict_class(double *y_pred, int num_predictions)
{
    for (int i = 0; i < num_predictions; i++)
    {
        y_pred[i] = round(y_pred[i]);
    }
}

// Calculate average loss using model-specific loss function
double average_loss(struct GeneralizedRegressor *genreg, double *y_true, double *y_pred, int num_predictions)
{
    double total_loss = 0.0;
    double p = genreg->p;

    for (int i = 0; i < num_predictions; i++)
    {
        total_loss += (p == -1) ? genreg->loss_function(y_true[i], y_pred[i]) : tweedie_nll(y_true[i], y_pred[i], p);
    }
    // Compute and return average loss
    return total_loss / num_predictions;
}
