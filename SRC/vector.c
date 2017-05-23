#include "pevsl_protos.h"

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

void vecset(int n, double t, double *v) {
  int i;
  for (i=0; i<n; i++) 
    v[i] = t; 
}

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

