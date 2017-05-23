//============================================================================
// Routines for computing eigenvalues of a symmetric tridiagonal matrix.
// They are wrappers of the LAPACK routine DSTEV() or sstev_()
//============================================================================

#include "pevsl_protos.h"

/**----------------------------------------------------------------------- 
 *  @brief compute all eigenvalues and eigenvectors of a symmetric tridiagonal
 *  matrix  
 *  @param n                The  dimension of the symmetric tridiagonal  matrix
 *  @param diag[],sdiag[]   Define the symmetric tridiagonal  matrix:  the
 *          diagonal elements are diag[0,...,n-1]  in order and the subdiagonal
 *          elements are sdiag[0,...,n-2] in order  
 *  @param[out] eigVal The output vector of length n containing all eigenvalues
 *          in ascending order 
 *  @param[out] eigVec The output n-by-n matrix with columns as eigenvectors,
 *          in the order as elements in eigVal. If NULL, then no eigenvector
 *          will be computed
 *  @return The flag returned by the
 *  LAPACK routine DSTEV() (if double  is double) or stev_() (if double
 *  is float)
 * --------------------------------------------------------------------- */

int SymmTridEig(double *eigVal, double *eigVec, int n, 
                const double *diag, const double *sdiag) {
  // compute eigenvalues and eigenvectors or eigvalues only
  char jobz = eigVec ? 'V' : 'N'; 
  int nn = n;
  int ldz = n;
  int info;  // output flag
  // copy diagonal and subdiagonal elements to alp and bet
  double *alp = eigVal;
  double *bet;
  PEVSL_MALLOC(bet, n-1, double);
  memcpy(alp, diag, n*sizeof(double));
  memcpy(bet, sdiag, (n-1)*sizeof(double));
  // allocate storage for computation
  double *sv = eigVec;
  double *work = NULL;
  if (jobz == 'V') {
    PEVSL_MALLOC(work, 2*n-2, double);
  }
  DSTEV(&jobz, &nn, alp, bet, sv, &ldz, work, &info);
  // free memory
  PEVSL_FREE(bet);
  if (work) {
    PEVSL_FREE(work);
  }
  if (info) {
    printf("DSTEV ERROR: INFO %d\n", info);
  }
  // return info
  return info;
}

/**----------------------------------------------------------------------- 
 *  @brief compute  eigenvalues and  eigenvectors of  a symmetric  tridiagonal
 *  matrix in a slice
 *  @param n The  dimension of  the  symmetric tridiagonal  matrix
 *  @param diag[],sdiag[]  define  the   symmetric  tridiagonal  matrix.  
 *  @param[out] eigVal Total number of eigenvalues found.
 *  @param[out] eigVec The first M elements contain teh selected eigenvalues in
 *  ascending oredr
 *  @param[in] vl If range='V', The lower bound of the interval to be searched
 *  for eigen values.
 *  @param[in] vu If  range='V', the upper bound of the interval to be searched
 *  for eigenvalues.
 *  @param[in] nevO If range='I', the index of the smallest eigen value to be
 *  returned.
 *
 *  This
 *  routine  computes selected  eigenvalues/vectors as  specified by  a
 *  range of values. This is a wrapper to the LAPACK routine DSTEMR().
 * ----------------------------------------------------------------------- */
int SymmTridEigS(double *eigVal, double *eigVec, int n, double vl, double vu,
                 int *nevO, const double *diag, const double *sdiag) {
  char jobz = 'V';  // compute eigenvalues and eigenvectors
  char range = 'V'; // compute eigenvalues in an interval

  // this does not use mwlapack for mex files
  int info;  
  //int idum = 0;
  //-------------------- isuppz not needed  
  int *isuppz;
  PEVSL_MALLOC(isuppz, 2*n, int);
  //-------------------- real work array
  double *work;
  int lwork = 18*n;
  PEVSL_MALLOC(work, lwork, double);
  //-------------------- int work array
  int *iwork;
  int liwork = 10*n;
  PEVSL_CALLOC(iwork, liwork, int);
  //-------------------- copy diagonal + subdiagonal elements
  //                     to alp and bet
  double *alp; 
  double *bet;
  PEVSL_MALLOC(bet, n, double);
  PEVSL_MALLOC(alp, n, double);
  //
  memcpy(alp, diag, n*sizeof(double));
  if (n > 1) {
    memcpy(bet, sdiag, (n-1)*sizeof(double));
  }

  //-------------------- allocate storage for computation
  logical tryrac = 1;
  double t0 = vl;
  double t1 = vu;

  DSTEMR(&jobz, &range, &n, alp, bet, &t0, &t1, NULL, NULL, nevO, 
         eigVal, eigVec, &n, &n, isuppz, &tryrac, work, &lwork, 
         iwork, &liwork, &info);

  if (info) {
    printf("dstemr_ error %d\n", info);
  }

  //-------------------- free memory
  PEVSL_FREE(bet);
  PEVSL_FREE(alp);
  PEVSL_FREE(work);
  PEVSL_FREE(iwork);
  PEVSL_FREE(isuppz);
  //
  return info;
}


/**- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *     @brief interface to   LAPACK SYMMETRIC EIGEN-SOLVER 
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
void SymEigenSolver(int n, double *A, int lda, double *Q, int ldq, double *lam) {
  /* compute eigenvalues/vectors of A that n x n, symmetric
   * eigenvalues saved in lam: the eigenvalues in ascending order
   * eigenvectors saved in Q */
  char jobz='V';/* want eigenvectors */
  char uplo='U';/* use upper triangular part of the matrix */
  /*   copy A to Q */
  int i,j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      Q[j+i*ldq] = A[j+i*lda];
    }
  }
  /*   first call: workspace query */
  double work_size;
  int lwork = -1, info;
  DSYEV(&jobz, &uplo, &n, Q, &ldq, lam, &work_size, &lwork, &info);
  if (info != 0) {
    fprintf(stdout, "DSYEV error [query call]: %d\n", info);
    exit(0);
  }
  /*   second call: do the computation */
  lwork = (int) work_size;
  double *work;
  PEVSL_MALLOC(work, lwork, double);
  DSYEV(&jobz, &uplo, &n, Q, &ldq, lam, work, &lwork, &info);
  if (info != 0) {
    fprintf(stdout, "DSYEV error [comp call]: %d\n", info);
    exit(0);
  }
  PEVSL_FREE(work);
}


/**
  * @brief Modified GS reortho with Daniel, Gragg, Kaufman, Stewart test 
 **/
void MGS_DGKS(int k, int i_max, pevsl_Parvec *Q, pevsl_Parvec *v, double *nrmv) {
  double eta = 1.0 / sqrt(2.0);
  double old_nrm, new_nrm;
  double w;
  int i, j;

  pEVSL_ParvecNrm2(v, &old_nrm);
  for (i=0; i<i_max; i++) {
    for (j=0; j<k; j++) {
      pEVSL_ParvecDot(&Q[j], v, &w);
      pEVSL_ParvecAxpy(-w, &Q[j], v);
    }
    pEVSL_ParvecNrm2(v, &new_nrm);
    if (new_nrm > eta * old_nrm) {
      break;
    }
    old_nrm = new_nrm;
  }
  if (nrmv) {
    *nrmv = new_nrm;
  }
}

/**
 * @brief Modified GS reortho. No test. just do i_max times
 * used in generalized ev problems
 * znew = z - (z, v_j)*z_j, for j=1,2,...
 **/
void MGS_DGKS2(int k, int i_max, pevsl_Parvec *Z, pevsl_Parvec *V, pevsl_Parvec *z) {
  double w;
  int i, j;
  for (i=0; i<i_max; i++) {
    for (j=0; j<k; j++) {
      pEVSL_ParvecDot(&V[j], z, &w);
      pEVSL_ParvecAxpy(-w, &Z[j], z);
    }
  }
}


