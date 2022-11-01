#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int* values;
    int* row;
    int nnz;
}csc_matrix;

#endif