#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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

// Computes log loss given true output and predicted probability
double log_loss(double y_true, double y_pred)
{
    return -y_true * log(fmax(y_pred, 1e-6)) - (1 - y_true) * log(1 - fmax(1 - y_pred, 1e-6));
}

// Computes log-likelihood given true output and predicted output
double log_likelihood(double y_true, double y_pred)
{
    return y_pred - y_true * log(fmax(y_pred, 1e-6));
}

// Computes negative log-likelihood given true output and predicted output
double neg_log_likelihood(double y_true, double y_pred)
{
    return -y_true * log(fmax(y_pred, 1e-6)) - y_pred;
}

// Performs a dot product on specified data entry
double dot_prod(double *X, int sample_index, double *w, int num_features)
{
    double product = 0.0;
    for (int j = 0; j < num_features; j++)
    {
        product += w[j] * X[sample_index * num_features + j];
    }
    return product;
}

// Reflexive identity link (Gaussian)
double identity_link(double eta)
{
    return eta;
}

// Logistic link (logit)
double logit_link(double eta)
{
    return log(eta / (1.0 - eta));
}

// Inverse logistic link
double sigmoid(double eta)
{
    return 1.0 / (1.0 + exp(-eta));
}

// Log-link
double log_link(double eta)
{
    return log(eta);
}

// Inverse log-link (exponential)
double inverse_log_link(double eta)
{
    return exp(eta);
}

// Inverse link
double inverse_link(double mu)
{
    return 1.0 / fmax(mu, 1e-6);
}

// Negative inverse link
double neg_inverse_link(double mu)
{
    return -1.0 / fmax(mu, 1e-6);
}

// Inverse of inverse link
double double_inverse_link(double eta)
{
    return 1.0 / fmax(eta, 1e-6); // Prevent undefined behavior
}

// Computes inverse Tweedie link using power parameter
double inverse_tweedie_link(double eta, double p)
{
    if (eta < 0 && (p > 1 && p < 2))
    {
        eta = 1e-6;
    }

    double exponent = 1.0 / (1.0 - p);
    exponent = fmax(exponent, -10); // Prevent undefined behavior

    return pow(fmax(eta, 1e-6), exponent);
}


// Computes half-deviance loss of Poisson distribution
double poisson_loss(double y_true, double y_pred)
{
    return y_true * log(y_true / y_pred) - (y_true - y_pred);
}

// Computes half-deviance loss of Gamma distribution
double gamma_loss(double y_true, double y_pred)
{
    return (y_true / y_pred) - log(y_true) - log(y_pred);
}

// Computes negative log-likelihood for Gaussian distributions (MSE)
double gaussian_nll(double y_true, double y_pred)
{
    double error = y_true - y_pred;
    return error * error;
}

// Computes negative log-likelihood for Poisson distributions
double poisson_nll(double y_true, double y_pred)
{
    return y_pred - y_true * log(y_pred);
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

// Computes R^2, a scale-independent performance metric, between predicted and actual values
double r_squared(double *y_true, double *y_pred, int num_predictions)
{
    // Calculate mean of actual values
    double y_mean = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        y_mean += y_true[i];
    }
    y_mean /= num_predictions;

    // Calculate sigma components
    double residuals_squared = 0.0;
    double mean_diff_squared = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        double residual = y_true[i] - y_pred[i];
        double mean_diff = y_true[i] - y_mean;

        residuals_squared += residual * residual;
        mean_diff_squared += mean_diff * mean_diff;
    }

    // Calculate R^2
    return 1.0 - (residuals_squared / mean_diff_squared);
}

// Computes Poisson deviance between predicted and actual values
double poisson_deviance(double *y_true, double *y_pred, int num_predictions)
{
    // Calculate mean of actual values
    double y_mean = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        y_mean += y_true[i];
    }
    y_mean /= num_predictions;

    // Calculate sigma model and null deviance components
    double deviance_model = 0.0;
    double deviance_null = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        double lambda_pred = fmax(y_pred[i], 1e-6); // Prevent log(0)
        double lambda_null = fmax(y_mean, 1e-6); // Prevent dividing by 0

        double yi = y_true[i];
        if (yi > 0)
        {
            deviance_model += 2.0 * (yi * log(yi / lambda_pred) - (yi - lambda_pred));
            deviance_null += 2.0 * (yi * log(yi / lambda_null) - (yi - lambda_null));
        }

    }

    // Prevent division by 0 if model is perfect fit
    if (deviance_null == 0.0)
    {
        return 1.0;
    }

    // Calculate Poisson deviance
    return 1.0 - (deviance_model / deviance_null);
}

// Computes deviance R^2 between predicted and actual values
double deviance_r2(double *y_true, double *y_pred, int num_predictions)
{
    // Calculate mean of actual values
    double y_mean = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        y_mean += y_true[i];
    }
    y_mean /= num_predictions;

    // Calculate sigma model and null deviance components
    double deviance_model = 0.0;
    double deviance_null = 0.0;
    for (int i = 0; i < num_predictions; i++) {
        double yi = fmax(y_true[i], 1e-6); // Prevent log(0) errors
        double mu = fmax(y_pred[i], 1e-6); // Prevent log(0) errors
        double mu_null = fmax(y_mean, 1e-6); // Prevent dividing by 0

        deviance_model += 2 * (yi * log(yi / mu) - (yi - mu));
        deviance_null += 2 * (yi * log(yi / mu_null) - (yi - mu_null));
    }

    // Calculate deviance R^2
    return 1.0 - (deviance_model / deviance_null);
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

// Computes average log-likelihood from actual output and predicted output
double average_log_likelihood(double *y_true, double *y_pred, int num_predictions)
{
    double total_loss = 0.0;

    for (int i = 0; i < num_predictions; i++)
    {
        total_loss += log_likelihood(y_true[i], y_pred[i]);
    }

    return (total_loss / num_predictions);
}

// Computes negative log-likelihood for Poisson distributions
double average_poisson_nll(double *y_true, double *y_pred, int num_predictions, int num_features)
{
    double total_loss = 0.0;
    for (int i = 0; i < num_predictions; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            total_loss += poisson_nll(y_true[i * num_features + j], y_pred[i * num_features + j]);
        }
    }
    // Calculate and return average loss
    return total_loss / num_predictions;
}

// Computes negative-log-likelihood for Tweedie distributions
double tweedie_nll(double y_true, double y_pred, double p)
{
    // Prevent computational errors
    y_pred = fmax(y_pred, 1e-6);

    if (y_true < 0.0)
    {
        printf("Predictor cannot be negative when using Tweedie loss\n");
        exit(EXIT_FAILURE);
    }

    // Compute Tweedie loss based on power parameter
    double loss = 0.0;
    if (p > 1.0 && p < 2.0)
    {
        if (y_true == 0.0)
        {
            loss = pow(y_pred, 2 - p) / (2 - p);
        }
        else
        {
            loss = pow(fmax(y_true, 1e-6), 2 - p) / (2 - p) - y_true * pow(y_pred, 1 - p) / (1 - p) + pow(y_pred, 2 - p) / (2 - p);
        }
    }
    return loss;
}

// Computes average Gamma loss from actual output and predicted output
double average_gamma_loss(double *y_true, double *y_pred, int num_predictions)
{
    double total_loss = 0.0;

    for (int i = 0; i < num_predictions; i++)
    {
        total_loss += (y_true[i] - y_pred[i]) / y_pred[i] - log(y_true[i] / y_pred[i]);
    }

    return total_loss / num_predictions;
}

// Computes residual for Gaussian mean-squared-error
double gaussian_residual(double y_true, double y_pred)
{
    return -1.0 * (y_true - y_pred);
}

// Computes residual for Poisson negative log-likelihood
double poisson_residual(double y_true, double y_pred)
{
    return -1.0 * (y_true - y_pred);
}

// Computes residual for Gamma negative log-likelihood
double gamma_residual(double y_true, double y_pred)
{
    return -(y_true - y_pred) / (y_pred * y_pred);
}

// Computes residual for Tweedie negative log-likelihood
double tweedie_residual(double y_true, double y_pred, double p)
{
    double derivative = (1.0 / (1.0 - p)) * pow(fmax(y_pred, 1e-6), fmin(p - 2, 10));
    return -(y_true - y_pred) * derivative;
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

// Print score/performance based on chosen model
void print_model_performance(char *distribution_type, double *y_true, double *y_pred, int num_predictions)
{
    // Mean-squared-error (MSE) and RMSE
    double mse = mean_squared_error(y_true, y_pred, num_predictions);
    printf("Test MSE: %g\n", mse);
    printf("Test RMSE: %g\n\n", sqrt(mse));

    if (strcmp(distribution_type, "gaussian") == 0 || strcmp(distribution_type, "normal") == 0)
    {
        printf("Test R^2: %g\n", r_squared(y_true, y_pred, num_predictions));
    }
    else if (strcmp(distribution_type, "poisson") == 0)
    {
        printf("Test Poisson Deviance: %g\n", poisson_deviance(y_true, y_pred, num_predictions));
    }
    else if (strcmp(distribution_type, "gamma") == 0 || strcmp(distribution_type, "tweedie") == 0)
    {
        printf("Test Deviance R^2: %g\n", deviance_r2(y_true, y_pred, num_predictions));
    }
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
double regularization_gradient(double weight, double l2_alpha, double l1_alpha, double r)
{
    double gradient = 0.0;

    // Check for zero regularization
    if (l2_alpha == 0.0 && l1_alpha == 0.0)
    {
        return gradient;
    }

    // Check for Elastic Net, adjusting coefficients accordingly
    if (l2_alpha > 0.0 && l1_alpha > 0.0)
    {
        l2_alpha = (1 - r) * l2_alpha;
        l1_alpha = r * l1_alpha;
    }

    // Add L2 regularization (ridge)
    if (l2_alpha > 0.0)
    {
        gradient += 2 * l2_alpha * weight;
    }

    // Add L1 regularization (lasso)
    if (l1_alpha > 0.0)
    {
        // Derivative of absolute value is dependent on sign of weight
        if (weight > 0)
        {
            gradient += l1_alpha;
        }
        else if (weight < 0)
        {
            gradient -= l1_alpha;
        }
    }
    return gradient;
}

// Computes total regularization loss
double regularization_loss(double weight, double l2_alpha, double l1_alpha, double r)
{
    double loss = 0.0;

    // Check for Elastic Net, adjusting coefficients accordingly
    if (l2_alpha > 0.0 && l1_alpha > 0.0)
    {
        l2_alpha = (1 - r) * l2_alpha;
        l1_alpha = r * l1_alpha;
    }

    // Add L2 regularization (ridge)
    if (l2_alpha > 0.0)
    {
        loss += l2_alpha * (weight * weight);
    }

    // Add L1 regularization (lasso)
    if (l1_alpha > 0.0)
    {
        loss += l1_alpha * fabs(weight);
    }

    return loss;
}
