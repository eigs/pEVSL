#include "pevsl_protos.h"

void dcsrgemv(char trans, int nrow, int ncol, double alp, double *a, 
              int *ia, int *ja, double *x, double bet, double *y);
void dcsrmv(char trans, int nrow, int ncol, double *a, 
            int *ia, int *ja, double *x, double *y);

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

void pEVSL_FreeCoo(pevsl_Coo *coo) {
  PEVSL_FREE(coo->ir);
  PEVSL_FREE(coo->jc);
  PEVSL_FREE(coo->vv);
}

int pEVSL_CooToCsr(int cooidx, pevsl_Coo *coo, pevsl_Csr *csr) {
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

