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

