#ifndef METRICS_H
#define METRICS_H

// Simple computation
double double_pow(double base, int pow);
double sigmoid(double z);
double log_loss(double y_true, double y_pred);

// Evaluation metrics
double mean_squared_error(double *y_true, double *y_pred, int num_predictions);
double average_log_loss(double *y_true, double *y_pred, int num_predictions);
double accuracy(double *y_true, double *y_pred, int num_predictions);
void classification_metrics(double *y_true, double *y_pred, int num_predictions, double *precision, double *recall, double *f1_score);

// Summation of total regularization gradient
double gradient_regularization(double weight, int train_length, double l2_alpha, double l1_alpha, double r);

#endif
