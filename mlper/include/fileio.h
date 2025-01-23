#ifndef FILEIO_H
#define FILEIO_H

#include "dataset.h"

void load(struct Dataset *data);
void export_predictions(double *y_pred, int num_predictions, char *file_name);
void export_results(struct Dataset *data, int num_predictions, int poly_degrees, char *file_name);

#endif
