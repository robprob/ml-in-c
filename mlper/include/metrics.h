#ifndef METRICS_H
#define METRICS_H

// Simple computation
double double_pow(double base, int pow);

// Evaluation metrics
double mean_squared_error(double *y_actual, double *y_pred, int num_predictions);

// Summation of total regularization gradient
double gradient_regularization(double weight, int train_length, double l2_alpha, double l1_alpha, double r);

#endif
