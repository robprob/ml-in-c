/*
Implementation of logistic regression.

Sigmoid Logistic Function
h(x) = 1 / (1 + e^-(wi * wj + b))

h(x): probabilistic prediction
wi: weight vector
xj: feature vector
b: bias parameter

Log Loss Function
E = −(1/N) * Σ[yi * log(h(xi)) + (1 - yi) * log(1 - h(xi))]
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "mlper.h"

struct LogisticRegression {
    int num_epochs;            // Number of training iterations
    int polynomial_degree;     // Always 1 for logistic regression
    double learning_rate;      // Training "step" size
    int early_stopping;        // Truth value of early stopping
    double tolerance;          // Minimum acceptable reduction in loss
    int patience;              // Number of "bad" loss checks required in a row
    int batch_size;            // Number of samples taken per epoch (leave at 0 for batch GD)
    double l2_alpha;           // Ridge coefficient
    double l1_alpha;           // Lasso coefficient
    double mix_ratio;          // 0 is equivalent to pure ridge, 1 is equivalent to pure lasso
    double *w;                 // Weight vector
    double b;                  // Bias
};

// Function Prototypes
void parse_config(struct Dataset *data, struct LogisticRegression *logreg);
void initialize_model(struct LogisticRegression *logreg, int num_features);
void fit_model(struct LogisticRegression *logreg, struct Dataset *data);
void print_model_parameters(struct LogisticRegression *logreg, struct Dataset *data);
void predict_prob(struct LogisticRegression *logreg, struct Dataset *data, double *X_predict, int num_predictions);
void predict_class(double *y_pred, int num_predictions);


int main(int argc, char **argv)
{
    clock_t start, end;

    start = clock();

    // Instantiate dataset at 0
    struct Dataset data = {0};
    // Instantiate Linear Regression model at 0
    struct LogisticRegression logreg = {0};

    // Parse data path and model hyperparameter from config file
    parse_config(&data, &logreg);

    /*
    // Print selected parameters
    printf("File Path: %s\n", data.file_path);
    printf("Standardized: %i\n", data.standardized);
    printf("Number of Epochs: %i\n", logreg.num_epochs);
    printf("Learning Rate: %g\n", logreg.learning_rate);
    printf("Test Proportion: %g\n", data.test_proportion);
    printf("Validation Proportion: %g\n", data.valid_proportion);
    printf("Early Stopping: %i\n", logreg.early_stopping);
    printf("Batch Size: %i\n", logreg.batch_size);
    printf("L2 alpha: %g\n", logreg.l2_alpha);
    printf("L1 alpha: %g\n", logreg.l1_alpha);
    printf("Elastic Net Mix Ratio: %g\n", logreg.mix_ratio);
    printf("Tolerance: %g\n", logreg.tolerance);
    printf("Patience %i\n", logreg.patience);
    */



    // Load feature and target variable data into arrays
    load(&data);
    // Transform feature matrix to specified highest degree
    poly_transform(&data, logreg.polynomial_degree);
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
    initialize_model(&logreg, data.num_features);

    clock_t train_start, train_end;

    train_start = clock();

    // Fit model to training data
    fit_model(&logreg, &data);

    train_end = clock();

    // Print trained model parameters, un-standardized if necessary
    print_model_parameters(&logreg, &data);

    // Calculate probabilities using trained model
    predict_prob(&logreg, &data, data.X_test, data.test_length);
    // Calculate Log Loss of test set
    double log_loss = average_log_loss(data.y_test, data.y_pred, data.test_length);
    printf("\nTest Log Loss: %f\n", log_loss);

    // Calculate class predictions using trained model
    predict_class(data.y_pred, data.test_length);

    // Calculate Accuracy
    double acc = accuracy(data.y_test, data.y_pred, data.test_length);
    printf("Test Accuracy: %f\n\n", acc);

    // Calculate precision, recall, F-1 score
    double precision = 0;
    double recall = 0;
    double f1_score = 0;
    classification_metrics(data.y_test, data.y_pred, data.test_length, &precision, &recall, &f1_score);
    printf("Test Precision: %f\n", precision);
    printf("Test Recall: %f\n", recall);
    printf("Test F1-Score: %f\n\n", f1_score);

    // Calculate and print model training time
    double training_time = ((double) (train_end - train_start)) / CLOCKS_PER_SEC;
    printf("Training Time: %f seconds\n", training_time);

    // Export feature data and calculated predictions
    export_results(&data, data.test_length, logreg.polynomial_degree, "test_predictions.csv");

    // Free memory taken up by dataset
    free_dataset(&data);
    // Free parameter weights array
    free(logreg.w);

    end = clock();
    double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Total CPU Time: %f seconds\n", total_time);
}

// Parse data file path and model hyperparameter from config file
void parse_config(struct Dataset *data, struct LogisticRegression *logreg)
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
            else if (strcmp(key, "standardize ") == 0)
            {
                data->standardized = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "num_epochs ") == 0)
            {
                logreg->num_epochs = atoi(value);
            }
            else if (strcmp(key, "learning_rate ") == 0)
            {
                logreg->learning_rate = atof(value);
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
                logreg->early_stopping = strcmp(value, "true") == 0 ? 1 : 0;
            }
            else if (strcmp(key, "batch_size ") == 0)
            {
                logreg->batch_size = atoi(value);
            }
            else if (strcmp(key, "l2_alpha ") == 0)
            {
                logreg->l2_alpha = atof(value);
            }
            else if (strcmp(key, "l1_alpha ") == 0)
            {
                logreg->l1_alpha = atof(value);
            }
            else if (strcmp(key, "mix_ratio ") == 0)
            {
                logreg->mix_ratio = atof(value);
            }
            else if (strcmp(key, "tolerance ") == 0)
            {
                logreg->tolerance = atof(value);
            }
            else if (strcmp(key, "patience ") == 0)
            {
                logreg->patience = atoi(value);
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

// Initialize model paramaters at 0
void initialize_model(struct LogisticRegression *logreg, int num_features)
{
    logreg->w = calloc(num_features, sizeof(double));
    if (!logreg->w)
    {
        printf("Unable to allocate memory for weights.\n");
        exit(EXIT_FAILURE);
    }
    logreg->b = 0.0;

    logreg->polynomial_degree = 1; // Always 1 for logistic regression
}

// Fit linear regression model to training data
void fit_model(struct LogisticRegression *logreg, struct Dataset *data)
{
    // Retrieve dataset counts and model parameters
    int num_features = data->num_features;
    int train_length = data->train_length;

    int num_epochs = logreg->num_epochs;
    double learning_rate = logreg->learning_rate;
    int batch_size = logreg->batch_size;
    double l2_alpha = logreg->l2_alpha;
    double l1_alpha = logreg->l1_alpha;
    double r = logreg->mix_ratio;

    double w[num_features];
    memcpy(w, logreg->w, num_features * sizeof(double));
    double b = logreg->b;

    // If necessary, implement validation error for early stopping
    int early_stopping = logreg->early_stopping;
    double valid_loss = 0.0;
    double prev_valid_loss = -1.0;
    double tolerance = logreg->tolerance;
    int patience = logreg->patience;
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
    printf("=========================================================\n");
    printf("| Epoch |  Train Loss  |   Accuracy   | Validation Loss |\n");
    printf("=========================================================\n");

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

        double total_loss = 0.0;
        int num_correct = 0;

        // Iterate training data, accumulating gradients
        for (int i = 0; i < train_length; i++) // "i" loops iterate training samples
        {
            // Calculate the linear combination, z
            double z = b;
            // Sum weighted feature contributions
            for (int j = 0; j < num_features; j++)
            {
                z += w[j] * data->X_train[i * num_features + j];
            }

            // Calculate output of sigmoid function
            double y_pred = sigmoid(z);

            // Compute log loss
            double y_true = data->y_train[i];
            total_loss += log_loss(y_true, y_pred);

            // Accumulate gradient of loss
            double error = y_pred - y_true;

            // For each feature
            for (int j= 0; j < num_features; j++)
            {
                w_sums[j] += data->X_train[i * num_features + j] * error;
            }
            b_sum += error;

            // Evaluate if prediction is correct
            if ((y_pred >= 0.5 && y_true == 1) || (y_pred < 0.5 && y_true == 0))
            {
                num_correct++;
            }
        }

        // Update weights and bias parameters
        for (int j = 0; j < num_features; j++)
        {
            double regularization = gradient_regularization(w[j], train_length, l2_alpha, l1_alpha, r);
            w[j] -= learning_rate * ((w_sums[j] / train_length) + regularization);
        }
        b -= learning_rate * (b_sum / train_length);

        // Compute average loss
        total_loss /= train_length;
        // Compute accuracy
        double accuracy = (double) num_correct / train_length;

        // Print training progress intermittently
        int divisor = (num_epochs / 10 == 0) ? 1 : num_epochs / 10; // Prevent dividing by 0 with small epoch number
        if (epoch % divisor == 0)
        {
            // Evaluate for validation set/early stopping
            if (data->valid_proportion == 0.0 || !early_stopping)
            {
                printf("| %5d | %12.5f | %11.2f%% |", epoch, total_loss, accuracy * 100);
                printf("       N/A       |\n");
                continue;
            }

            // Copy weights and bias parameters back to model
            // Final weights and bias update
            memcpy(logreg->w, w, num_features * sizeof(double));
            logreg->b = b;

            // Calculate probabilities and calculate average loss of validation set
            predict_prob(logreg, data, data->X_valid, data->valid_length);
            valid_loss = average_log_loss(data->y_valid, data->y_pred, data->valid_length);

            printf("| %5d | %12.5f | %11.2f%% | %15.5f |\n", epoch, total_loss, accuracy * 100, valid_loss);

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
                    printf("---------------------------------------------------------\n");
                    printf("Stopping early at epoch %d, validation set has reached a minimum loss.\n", epoch);
                    free(w_sums);
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
    memcpy(logreg->w, w, num_features * sizeof(double));
    logreg->b = b;

    // Print footer of terminal log
    printf("---------------------------------------------------------\n");

    // Free gradient accumulator array
    free(w_sums);
}

// Print trained weight and bias parameters, un-standardized if necessary
void print_model_parameters(struct LogisticRegression *logreg, struct Dataset *data)
{
    int num_features = data->num_features;

    printf("\nWeights:\n");
    for (int j = 0; j < num_features; j++)
    {
        printf("    Feature %3d: ",j + 1);

        // Un-standardize model weight, if necessary
        double weight = logreg->w[j];
        if (data->standardized)
        {
            weight /= data->feature_stds[j];
        }

        // Print weight
        printf(" %6.3f\n", weight);
    }

    // Un-standardize bias, if necessary
    double bias = logreg->b;
    if (data->standardized)
    {
        for (int j = 0; j < num_features; j++)
        {
            // Parse original data mean and standard deviation
            double feature_mean = data->feature_means[j];
            double feature_std = data->feature_stds[j];
            bias -= (logreg->w[j] * feature_mean) / feature_std;
        }
    }
    printf("Bias: %.3f\n", bias);
}

// Make probability predictions using trained logistic regression model
void predict_prob(struct LogisticRegression *logreg, struct Dataset *data, double *X_predict, int num_predictions)
{
    int num_features = data->num_features;

    // Reallocate y_pred array for size of predictions
    data->y_pred = realloc(data->y_pred, num_predictions * sizeof(double));
    if (!data->y_pred)
    {
        printf("Unable to reallocate y_pred.\n");
        exit(EXIT_FAILURE);
    }

    double b = logreg->b;

    // Calculate and store predictions
    for (int i = 0; i < num_predictions; i++)
    {
        double z = b;
        // Sum weighted feature contributions
        for (int j = 0; j < num_features; j++)
        {
            z += logreg->w[j] * X_predict[i * num_features + j];
        }
        data->y_pred[i] = sigmoid(z);
    }
}

// Convert existing probability predictions into discrete predictions
void predict_class(double *y_pred, int num_predictions)
{
    for (int i = 0; i < num_predictions; i++)
    {
        if (y_pred[i] >= 0.5)
        {
            y_pred[i] = 1;
        }
        else
        {
            y_pred[i] = 0;
        }
    }
}
