#include "pevsl_int.h"

void pEVSL_CsrResize(int nrow, int ncol, int nnz, pevsl_Csr *csr) {
  csr->nrows = nrow;
  csr->ncols = ncol;
  PEVSL_MALLOC(csr->ia, nrow+1, int);
  PEVSL_MALLOC(csr->ja, nnz, int);
  PEVSL_MALLOC(csr->a, nnz, double);
}

void pEVSL_FreeCsr(pevsl_Csr *csr) {
  PEVSL_FREE(csr->ia);
  PEVSL_FREE(csr->ja);
  PEVSL_FREE(csr->a);
}

void pEVSL_CooResize(int nrow, int ncol, int nnz, pevsl_Coo *coo) {
  coo->nrows = nrow;
  coo->ncols = ncol;
  coo->nnz = nnz;
  PEVSL_MALLOC(coo->ir, nnz, int);
  PEVSL_MALLOC(coo->jc, nnz, int);
  PEVSL_MALLOC(coo->vv, nnz, double);
}

void pEVSL_FreeCoo(pevsl_Coo *coo) {
  PEVSL_FREE(coo->ir);
  PEVSL_FREE(coo->jc);
  PEVSL_FREE(coo->vv);
}

int pEVSL_CooToCsr(int cooidx, pevsl_Coo *coo, pevsl_Csr *csr) {
  cooidx = cooidx != 0;
  int nnz = coo->nnz;
  pEVSL_CsrResize(coo->nrows, coo->ncols, nnz, csr);
  int i;
  for (i=0; i<coo->nrows+1; i++) {
    csr->ia[i] = 0;
  }
  for (i=0; i<nnz; i++) {
    int row = coo->ir[i] - cooidx;
    csr->ia[row+1] ++;
  }
  for (i=0; i<coo->nrows; i++) {
    csr->ia[i+1] += csr->ia[i];
  }
  for (i=0; i<nnz; i++) {
    int row = coo->ir[i] - cooidx;
    int col = coo->jc[i] - cooidx;
    double val = coo->vv[i];
    int k = csr->ia[row];
    csr->a[k] = val;
    csr->ja[k] = col;
    csr->ia[row]++;
  }
  for (i=coo->nrows; i>0; i--) {
    csr->ia[i] = csr->ia[i-1];
  }
  csr->ia[0] = 0;

  return 0;
}

int pEVSL_CsrToCoo(pevsl_Csr *csr, int cooidx, pevsl_Coo *coo) {
  cooidx = cooidx != 0;
  int nnz = PEVSL_CSRNNZ(csr);
  pEVSL_CooResize(csr->nrows, csr->ncols, nnz, coo);
  int i,j,k=0;
  for (i=0; i<csr->nrows; i++) {
    for (j=csr->ia[i]; j<csr->ia[i+1]; j++) {
      coo->ir[k] = i + cooidx;
      coo->jc[k] = csr->ja[j] + cooidx;
      coo->vv[k] = csr->a[j];
      k++;
    }
  }
  PEVSL_CHKERR(k != nnz);
  return 0;
}

void dcsrgemv(char trans, int nrow, int ncol, double alp, double *a, 
              int *ia, int *ja, double *x, double bet, double *y) {
  int i,j;
  if (trans == 'N') {
    for (i=0; i<nrow; i++) {
      double r = 0.0;
      for (j=ia[i]; j<ia[i+1]; j++) {
        r += a[j] * x[ja[j]];
      }
      y[i] = bet*y[i] + alp*r;
    }
  } else {  
    for (i=0; i<ncol; i++) {
      y[i] *= bet;
    }
    for (i=0; i<nrow; i++) {
      double xi = alp * x[i];
      for (j=ia[i]; j<ia[i+1]; j++) {
        y[ja[j]] += a[j] * xi;
      }
    }
  }
}

// y = alp*A*x + bet*y
int pEVSL_MatvecGen(double alp, pevsl_Csr *A, double *x, double bet, double *y) {
  dcsrgemv('N', A->nrows, A->ncols, alp, A->a, A->ia, A->ja, x, bet, y);

  return 0;
}

void dcsrmv(char trans, int nrow, int ncol, double *a, 
            int *ia, int *ja, double *x, double *y) {
  int i,j;
  if (trans == 'N') {  
    for (i=0; i<nrow; i++) {
      double r = 0.0;
      for (j=ia[i]; j<ia[i+1]; j++) {
        r += a[j] * x[ja[j]];
      }
      y[i] = r;
    }
  } else {  
    for (i=0; i<ncol; i++) {
      y[i] = 0;
    }
    for (i=0; i<nrow; i++) {
      double xi = x[i];
      for (j=ia[i]; j<ia[i+1]; j++) {
        y[ja[j]] += a[j] * xi;
      }
    }
  }
}

// y = A * x
int pEVSL_Matvec(pevsl_Csr *A, double *x, double *y) {
  dcsrmv('N', A->nrows, A->ncols, A->a, A->ia, A->ja, x, y);

  return 0;
}

/**
 * @brief convert csr to csc
 * Assume input csr is 0-based index
 * output csc 0/1 index specified by OUTINDEX      *
 */
void csrcsc(int OUTINDEX, int nrow, int ncol, int job,
            double *a, int *ja, int *ia,
            double *ao, int *jao, int *iao) {
  int i,k;
  for (i=0; i<ncol+1; i++) {
    iao[i] = 0;
  }
  // compute nnz of columns of A
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      iao[ja[k]+1] ++;
    }
  }
  // compute pointers from lengths
  for (i=0; i<ncol; i++) {
    iao[i+1] += iao[i];
  }
  // now do the actual copying
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      int j = ja[k];
      if (job) {
        ao[iao[j]] = a[k];
      }
      jao[iao[j]++] = i + OUTINDEX;
    }
  }
  /*---- reshift iao and leave */
  for (i=ncol; i>0; i--) {
    iao[i] = iao[i-1] + OUTINDEX;
  }
  iao[0] = OUTINDEX;
}

/**
 * @brief  Sort each row of a csr by increasing column 
 * order
 * By double transposition
 */
void pEVSL_SortRow(pevsl_Csr *A) {
  /*-------------------------------------------*/
  int nrows = A->nrows;
  int ncols = A->ncols;
  int nnz = A->ia[nrows];
  // work array
  double *b;
  int *jb, *ib;
  PEVSL_MALLOC(b, nnz, double);
  PEVSL_MALLOC(jb, nnz, int);
  PEVSL_MALLOC(ib, ncols+1, int);
  // double transposition
  csrcsc(0, nrows, ncols, 1, A->a, A->ja, A->ia, b, jb, ib);
  csrcsc(0, ncols, nrows, 1, b, jb, ib, A->a, A->ja, A->ia);
  // free
  PEVSL_FREE(b);
  PEVSL_FREE(jb);
  PEVSL_FREE(ib);
}

