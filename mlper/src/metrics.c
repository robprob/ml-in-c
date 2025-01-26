#include <math.h>

// Computes specified power of a double as a lightweight alternative to Math.pow
double double_pow(double base, int pow)
{
    double result = 1.0;
    for (int i = 0; i < pow; i++)
    {
        result *= base;
    }
    return result;
}

// Computes sigmoid output given exponent, z
double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

// Computes log loss given true output and predicted probability
double log_loss(double y_true, double y_pred)
{
    return -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);
}

// Computes Mean Squared Error (MSE) between predicted and actual values
double mean_squared_error(double *y_true, double *y_pred, int num_predictions)
{
    // Sum of absolute error squared
    double sum_error = 0;
    // Iterate predictions
    for (int i = 0; i < num_predictions; i++)
    {
        // Calculate error
        double abs_error = y_true[i] - y_pred[i];
        // Add squared result to sum of error
        sum_error += abs_error * abs_error;
    }

    // Return average sum of absolute error squared
    return (sum_error / num_predictions);
}

// Computes average log loss from actual output and predicted probabilities
double average_log_loss(double *y_true, double *y_pred, int num_predictions)
{
    double total_loss = 0.0;

    for (int i = 0; i < num_predictions; i++)
    {
        total_loss += log_loss(y_true[i], y_pred[i]);
    }

    return (total_loss / num_predictions);
}

// Computes overall accuracy of discrete predictions (float in range [0.0, 1.0] inclusive)
double accuracy(double *y_true, double *y_pred, int num_predictions)
{
    double num_correct = 0.0;

    for (int i = 0; i < num_predictions; i++)
    {
        if (y_pred[i] == y_true[i])
        {
            num_correct++;
        }
    }

    return (double) num_correct / num_predictions;
}

// Computes precision, recall, and f1-score
void classification_metrics(double *y_true, double *y_pred, int num_predictions, double *precision, double *recall, double *f1_score)
{
    int tp = 0; // True positives
    int fp = 0; // False positives
    int fn = 0; // False negatives

    for (int i = 0; i < num_predictions; i++)
    {
        if (y_pred[i] == y_true[i])
        {
            tp++;
        }
        else if (y_pred[i] == 1 && y_true[i] == 0)
        {
            fp++;
        }
        else if (y_pred[i] == 0 && y_true[i] == 1)
        {
            fn++;
        }
    }

    if (tp + fp == 0)
    {
        *precision = 0;
    }
    else
    {
        *precision = (double) tp / (tp + fp);
    }

    if (tp + fn == 0)
    {
        *recall = 0;
    }
    else
    {
        *recall = (double) tp / (tp + fn);
    }

    if (*precision + *recall == 0)
    {
        *f1_score = 0;
    }
    else
    {
        *f1_score = 2 * (*precision * *recall) / (*precision + *recall);
    }
}

// Computes total regularization gradient penalty
double gradient_regularization(double weight, int train_length, double l2_alpha, double l1_alpha, double r)
{
    double gradient = 0.0;

    // Check for Elastic Net, adjusting coefficients accordingly
    if (l2_alpha > 0.0 && l1_alpha > 0.0)
    {
        l2_alpha = (1 - r) * l2_alpha;
        l1_alpha = r * l1_alpha;
    }

    // Add L2 regularization (ridge)
    if (l2_alpha > 0.0)
    {
        gradient += (2.0 / train_length) * l2_alpha * weight;
    }

    // Add L1 regularization (lasso)
    if (l1_alpha > 0.0)
    {
        // Derivative of absolute value is dependent on sign of weight
        if (weight > 0)
        {
            gradient+= l1_alpha;
        }
        else if (weight < 0)
        {
            gradient -= l1_alpha;
        }
    }

    return gradient;
}

