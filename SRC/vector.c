#include "pevsl_int.h"

/**
 * @file vector.c
 * @brief Vector related functions
 * */
/**
 * Linearly partitions an interval (linspace in Matlab)
 * @param[in] left end of interval
 * @param[in] right end of interval
 * @param[in] num Number of points
 * @param[out] arr Linearly spaced points
 * 
 */
void linspace(double a, double b, int num, double *arr) {
  double h;
  h = num == 1 ? 0 : (b-a)/(num-1);
  int i;
  //-------------------- careful at the boundaries!
  arr[0] = a;
  arr[num-1] = b;
  for (i=1; i<num-1; i++) {
    arr[i] = a+i*h;
  }
}

/**
 * Sets all elements of v to t
 * @param[in] n Number of elements
 * @param[in] t Value which elements should be set to
 * @param[out] v Vector to set
 * */
void vecset(int n, double t, double *v) {
  int i;
  for (i=0; i<n; i++) 
    v[i] = t; 
}

/** 
 * @brief Compares a,b as doubles
 * @param[in] a First value
 * @param[in] b Second value
 * @return -1 if b>a, 0 if a==b, 1 otherwise
 * */
int compare1(const void *a, const void *b) {
  double *aa = (double*) a;
  double *bb = (double*) b;
  if (*aa < *bb) {
    return -1;
  } else if (*aa == *bb) {
    return 0;
  } else {
    return 1;
  }
}
typedef struct _doubleint {
  int i;
  double d;
} doubleint;

/** 
 * @brief Compares the doubles of a,b as double/int pairs
 * @param[in] a First value
 * @param[in] b Second value
 * @return -1 if b>a, 0 if a==b, 1 otherwise
 * */
int compare2(const void *a, const void *b) {
  const doubleint *aa = (doubleint*) a;
  const doubleint *bb = (doubleint*) b;
  if (aa->d < bb->d) {
    return -1;
  } else if (aa->d == bb->d) {
    return 0;
  } else {
    return 1;
  }
}
/** 
 * @brief Sorts a vector, and potentially indices
 * @param[in] n Number of elements
 * @param[in, out] v Vector to sort
 * @param[in, out] ind Indices to sort
 *
 * */
void sort_double(int n, double *v, int *ind) {
  /* if sorting indices are not wanted */
  if (ind == NULL) {
    qsort(v, n, sizeof(double), compare1);
    return;
  }
  doubleint *vv;
  PEVSL_MALLOC(vv, n, doubleint);
  int i;
  for (i=0; i<n; i++) {
    vv[i].d = v[i];
    vv[i].i = i;
  }
  qsort(vv, n, sizeof(doubleint), compare2);
  for (i=0; i<n; i++) {
    v[i] = vv[i].d;
    ind[i] = vv[i].i;
  }
  PEVSL_FREE(vv);
}

