#ifndef METRICS_H
#define METRICS_H

// Simple computation
double double_pow(double base, int pow);
double log_loss(double y_true, double y_pred);
double log_likelihood(double y_true, double y_pred);
double neg_log_likelihood(double y_true, double y_pred);

// Linear algebra
double dot_prod(double *X, int sample_index, double *w, int num_features);

// Link and reverse link functions
double identity_link(double mu);
double logit_link(double mu);
double sigmoid(double eta);
double log_link(double mu);
double inverse_log_link(double eta);
double inverse_link(double mu);
double neg_inverse_link(double mu);
double double_inverse_link(double eta);
double inverse_tweedie_link(double eta, double p);

// Loss functions
double poisson_loss(double y_true, double y_pred);
double gamma_loss(double y_true, double y_pred);
double gaussian_nll(double y_true, double y_pred);
double poisson_nll(double y_true, double y_pred);
double tweedie_nll(double y_true, double y_pred, double p);
double average_log_loss(double *y_true, double *y_pred, int num_predictions);
double average_log_likelihood(double *y_true, double *y_pred, int num_predictions);
double average_poisson_nll(double *y_true, double *y_pred, int num_predictions, int num_features);
double average_gamma_loss(double *y_true, double *y_pred, int num_predictions);

// Residuals of derivative loss functions, for use in optimization
double gaussian_residual(double y_true, double y_pred);
double poisson_residual(double y_true, double y_pred);
double gamma_residual(double y_true, double y_pred);
double tweedie_residual(double y_true, double y_pred, double p);

// Evaluation metrics
double accuracy(double *y_true, double *y_pred, int num_predictions);
double mean_squared_error(double *y_true, double *y_pred, int num_predictions);
double r_squared(double *y_true, double *y_pred, int num_predictions);
double poisson_deviance(double *y_true, double *y_pred, int num_predictions);
double deviance_r2(double *y_true, double *y_pred, int num_predictions);
void print_model_performance(char *distribution, double *y_true, double *y_pred, int num_predictions);
void classification_metrics(double *y_true, double *y_pred, int num_predictions, double *precision, double *recall, double *f1_score);

// Regularization
double regularization_gradient(double weight, double l2_alpha, double l1_alpha, double r);
double regularization_loss(double weight, double l2_alpha, double l1_alpha, double r);

#endif
