#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "dataset.h"

// Feature transformation
void poly_transform(struct Dataset *data, int degree);

// Feature scaling
void standardize(struct Dataset *data);
void unstandardize(struct Dataset *data, double *feature_data, int num_samples);

#endif
