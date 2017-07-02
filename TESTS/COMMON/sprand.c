#include <stdlib.h>
#include "pevsl.h"
#include "common.h"

/* randomly pick m elements from [0,1,...,n-1]
 * when n == m, this is to generate a random permutation of length n
 * by Fisher–Yates shuffle alg
 * Input: elem: arry of length n. on entry: must be a valid permutation
 * of [0,...,n-1]
 * Output: elem: on exit the first m entries are the ones we want
 * and elem will be changed to some permutation */
int RandElems(int n, int m, int *elem) {
  int i,j,k;
  if (m > n) {
    return 1;
  }
  /* do m exchanges */
  for (i=0; i<m; i++) {
    /* j = random number in [i,...,n-1] */
    j = i + (rand() % (n-i));
    /* exchange p[j] and p[i] */
    if (i != j) {
      k = elem[j];
      elem[j] = elem[i];
      elem[i] = k;
    }
  }
  return 0;
}

/* @brief create a random sparse matrix
 * nrow: number of rows
 * ncol: number of columns
 * rownnz: number of nonzeros per row
 * csr: the matrix
 */
void SpRandCsr(int nrow, int ncol, int rownnz, pevsl_Csr *csr) {
  int i, j, cnt, *elem, ierr;
  /* at most ncol nnz per row */
  rownnz = PEVSL_MIN(ncol, rownnz);
  /* array for random selection */
  PEVSL_MALLOC(elem, ncol, int);
  for (i=0; i<ncol; i++) {
    elem[i] = i;
  }
  /* allocate CSR */
  pEVSL_CsrResize(nrow, ncol, nrow*rownnz, csr);
  csr->ia[0] = 0;
  for (i=0, cnt=0; i<nrow; i++) {
    /* row i, randomly pick rownnz elements */
    ierr = RandElems(ncol, rownnz, elem);
    PEVSL_CHKERR(ierr);
    /* copy them to csr */
    for (j=0; j<rownnz; j++) {
      csr->ja[cnt] = elem[j];
      double t = rand() / ((double) RAND_MAX);
      csr->a[cnt++] = t;
    }
    /* next row ptr */
    csr->ia[i+1] = cnt;
  }
  PEVSL_CHKERR(cnt != nrow*rownnz);
  PEVSL_FREE(elem);
}

