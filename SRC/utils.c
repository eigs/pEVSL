#include "pevsl_protos.h"

typedef struct _doubleint {
  int i;
  double d;
} doubleint;

int compareInt(const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

int compareDouble(const void *a, const void *b) {
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

int compareDoubleInt(const void *a, const void *b) {
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

void pEVSL_SortInt(int *x, int n) {
    qsort(x, n, sizeof(int), compareInt);
}

void pEVSL_SortDouble(int n, double *v, int *ind) {
  /* if sorting indices are not wanted */
  if (ind == NULL) {
    qsort(v, n, sizeof(double), compareDouble);
    return;
  }
  doubleint *vv;
  PEVSL_MALLOC(vv, n, doubleint);
  int i;
  for (i=0; i<n; i++) {
    vv[i].d = v[i];
    vv[i].i = i;
  }
  qsort(vv, n, sizeof(doubleint), compareDoubleInt);
  for (i=0; i<n; i++) {
    v[i] = vv[i].d;
    ind[i] = vv[i].i;
  }
  PEVSL_FREE(vv);
}

void pEVSL_LinSpace(double a, double b, int num, double *arr) {
  double h;
  h = (num==1? 0: (b-a)/(num-1));
  int i;
 //-------------------- careful at the boundaries!
  arr[0] = a;
  arr[num-1] = b;
  for (i=1; i<num-1; i++)
    arr[i] = a+i*h;

}

void pEVSL_Part1d(int len, int pnum, int *idx, int *j1, int *j2, int job) {
/*-----------------------------------------*
   Partition of 1D array
   Input:
   len:  length of the array
   pnum: partition number
   idx [job=1]:  index of a partition
   j1 [job=2]: index of an entry
   Output: range of this partition
   j1,j2 [job=1]: partition [j1, j2)
   idx [job=2]: partition index
   j2 [job=2]: size of this partition

Example: partition  9 into 4 parts: [3,2,2,2]
         partition 10 into 4 parts: [3,3,2,2]
         partition 11 into 4 parts: [3,3,3,2]
 *-----------------------------------------*/
    int size = (len+pnum-1)/pnum;
    int cc = pnum - (size*pnum - len);
    if (job == 1) {
        if (*idx < cc) {
            *j1 = *idx * size;
        } else {
            *j1 = len - (--size)*(pnum - *idx);
        }
        *j2 = *j1 + size;
    } else {
        if (*j1 < size*cc) {
            *idx = *j1 / size;
        } else {
            *idx = cc + (*j1-size*cc)/(size-1);
            --size;
        }
        if (j2) {
            *j2 = size;
        }
    }
}

int pEVSL_BinarySearch(int *x, int n, int key) {
    int *p = bsearch(&key, x, n, sizeof(int), compareInt);
    if (p) {
        return (p - x);
    } else {
        // not found
        return -1;
    }
}

void pEVSL_Vecset(int n, double t, double *v) {
  int i;
  for (i=0; i<n; i++) 
    v[i] = t; 
}


