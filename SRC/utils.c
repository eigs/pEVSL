#include "pevsl_int.h"
/**
 * @file utils.c
 * @brief Utility functions
 * */

typedef struct _doubleint {
  int i;
  double d;
} doubleint;

int compareInt(const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

/**
 * @brief Compares a,b as doubles
 * @param[in] a First value
 * @param[in] b Second value
 * @return -1 if b>a, 0 if a==b, 1 otherwise
 * */
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

/**
 * @brief Compares the doubles of a,b as double/int pairs
 * @param[in] a First value
 * @param[in] b Second value
 * @return -1 if b>a, 0 if a==b, 1 otherwise
 * */
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

/**
 * @brief Sorts a vector of ints, and potentially indices
 * @param[in] n Number of elements
 * @param[in, out] x Vector to sort
 *
 * */
void pEVSL_SortInt(int *x, int n) {
    qsort(x, n, sizeof(int), compareInt);
}

/**
 * @brief Sorts a vector, and potentially indices
 * @param[in] n Number of elements
 * @param[in, out] v Vector to sort
 * @param[in, out] ind Indices to sort
 *
 * */
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

/**
 * Linearly partitions an interval (linspace in Matlab)
 * @param[in] a left end of interval
 * @param[in] b right end of interval
 * @param[in] num Number of points
 * @param[out] arr Linearly spaced points
 *
 */
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

/**-----------------------------------------*
 * @brief
   Partition of 1D array
   Input:
   @param[in] len  length of the array
   @param[in] pnum partition number
   @param[in] job Flag
   @param[in] idx [job=1]:  index of a partition
   @param[in] j1 [job=2]: index of an entry
   Output: range of this partition
   @param[out] j1,j2 [job=1]: partition [j1, j2)
   @param[out] idx [job=2]: partition index
   @param[out] j2 [job=2]: size of this partition

Example: partition  9 into 4 parts: [3,2,2,2]
         partition 10 into 4 parts: [3,3,2,2]
         partition 11 into 4 parts: [3,3,3,2]
 *-----------------------------------------*/
void pEVSL_Part1d(int len, int pnum, int *idx, int *j1, int *j2, int job) {
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

/*! @brief search an element in a sorted array,
 *  @param[in] x vector
 *  @param[in] n number of elements
 *  @param[in] key key
 *  @return
 *  if found return its position
 *  if not found, return -1
 */
int pEVSL_BinarySearch(int *x, int n, int key) {
    int *p = bsearch(&key, x, n, sizeof(int), compareInt);
    if (p) {
        return (p - x);
    } else {
        // not found
        return -1;
    }
}
/*! search an element in a sorted array, of length n,
 *  that represents intervals
 *  [x0, x1), [x1, x2), [x2, x3),...,[x_{n-2}, x_{n-1})
 *  @param[in] x vector
 *  @param[in] n number of elements
 *  @param[in] key key
 *  @return
 *  if x_{i} <= key < x_{i+1}, return i
 *  if key < x_{0}, return -1
 *  if x_{n-1} <= key return n-1
 */
int pEVSL_BinarySearchInterval(int *x, int n, int key) {
  if (key < x[0]) {
    return -1;
  }
  if (key >= x[n-1]) {
    return n-1;
  }
  int a = 0;
  int b = n-2;
  while (a < b) {
    int c = (a+b)/2;
    if (key >= x[c] && key < x[c+1]) {
      return c;
    } else if (key < x[c]) {
      b = c-1;
    } else {
      a = c+1;
    }
  }
  return a;
}

/**
 * @brief sets a vector (double array) to a constant
 * @param[in] n number of elements
 * @param[in] t constant
 * @param[in] v vector
 */
void pEVSL_Vecset(int n, double t, double *v) {
  int i;
  for (i=0; i<n; i++)
    v[i] = t;
}


double pEVSL_Wtime() {
  return MPI_Wtime();
}

