/*
  This file is the internal header file of pESL that contains data structures, 
  constant definitions, and internal function prototypes 
*/

#ifndef PEVSL_INT_H
#define PEVSL_INT_H

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <float.h>

#include "pevsl_struct.h"
#include "pevsl_def.h"
#include "pevsl.h"

/* BLAS LAPACK */
#define DCOPY    dcopy_
#define DDOT     ddot_
#define DNRM2    dnrm2_
#define DSCAL    dscal_
#define DASUM    dasum_
#define DGEMV    dgemv_
#define DGEMM    dgemm_
#define DAXPY    daxpy_
#define DSTEV    dstev_
#define DSYEV    dsyev_
#define DSTEMR   dstemr_
#define DHSEQR   dhseqr_
#define ZGESV    zgesv_

typedef int logical;
void DCOPY(int *n, double *dx, int *incx, double *dy, int *incy);
void DAXPY(int *n,double *alpha,double *x,int *incx,double *y,int *incy);
void DSCAL(int *n,double *a,double *x,int *incx);
double DASUM(int *n,double *x,int *incx);
double DDOT(int *n,double *x,int *incx,double *y,int *incy);
double DNRM2(int *n,double *x,int *incx);
void DGEMM(char *transa,char *transb,int *m,int *n,int *k,double *alpha,
           double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc);
void DGEMV(char *trans, int *m, int *n, double *alpha, double *a, int *lda, 
           double *x, int *incx, double *beta, double *y, int *incy);
void DSTEV(char *jobz, int *n, double *diagonal, double *subdiagonal, 
           double *V, int *ldz, double *work, int *info);
void DSYEV(char* jobz,char* uplo,int* n,double* fa,int* lda,double* w, 
           double* work,int* lwork,int* info);
void DSTEMR(char *jobz, char *range, int *n, double *D, double *E, double *VL, 
            double *VU, int *IL, int *IU, int *M, double *W, double *Z, 
            int *LDZ, int *NZC, int *ISUPPZ, logical *TRYRAC, double *WORK, 
	    int *LWORK, int *IWORK, int *LIWORK, int *INFO);
void DHSEQR(char* jobz, char* compz, int* n, int* ilo, int* ihi, double* h,
            int* ldh, double* wr, double* wi, double* z, int* ldz, double* work, 
            int* lwork,int* info);
void ZGESV(int *n, int *nrow, complex double * A, int* m, int* ipiv, 
           complex double *rhs, int* k, int* INFO);

/* chebpol.c */
int pEVSL_ChebAv(pevsl_Polparams *pol, pevsl_Parvec *v, pevsl_Parvec *y, pevsl_Parvec *w);

/* miscla.c */
int SymmTridEig(double *eigVal, double *eigVec, int n, const double *diag, const double *sdiag);
int SymmTridEigS(double *eigVal, double *eigVec, int n, double vl, double vu,
                 int *nevO, const double *diag, const double *sdiag);
void SymEigenSolver(int n, double *A, int lda, double *Q, int ldq, double *lam);
void CGS_DGKS(int k, int i_max, pevsl_Parvecs *Q, pevsl_Parvec *v, double *nrmv, double *w);
void CGS_DGKS2(int k, int i_max, pevsl_Parvecs *V, pevsl_Parvecs *Z, pevsl_Parvec *v,
               double *w);

/* parcsrmv.c */
void pEVSL_ParcsrMatvec0(double *x, double *y, void *data);

/* utils.c */
int pEVSL_BinarySearchInterval(int *x, int n, int key);
double pEVSL_Wtime();

/*------------------- inline functions */
/** 
 * @brief fprintf, only for rank == 0
 * 
 * */
static inline void pEVSL_fprintf0(int rank, FILE *fp, const char *format, ...) {
  if (rank) {
    return;
  }
  va_list args;
  va_start(args, format);
  vfprintf(fp, format, args);
  va_end(args);
}

/** 
 * @brief Perform matrix-vector product y = A * x
 * 
 * */
static inline void pEVSL_MatvecA(pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(!pevsl_data.Amv);
     
  PEVSL_CHKERR(pevsl_data.N != x->n_global);
  PEVSL_CHKERR(pevsl_data.n != x->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != x->n_first);
  PEVSL_CHKERR(pevsl_data.N != y->n_global);
  PEVSL_CHKERR(pevsl_data.n != y->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != y->n_first);

  double tms = pEVSL_Wtime();

  pevsl_data.Amv->func(x->data, y->data, pevsl_data.Amv->data);
  
  double tme = pEVSL_Wtime();
  pevsl_stat.t_mvA += tme - tms;
  pevsl_stat.n_mvA ++;
}


/** 
 * @brief Perform matrix-vector product y = B * x
 * 
 * */
static inline void pEVSL_MatvecB(pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(!pevsl_data.Bmv);
  
  PEVSL_CHKERR(pevsl_data.N != x->n_global);
  PEVSL_CHKERR(pevsl_data.n != x->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != x->n_first);
  PEVSL_CHKERR(pevsl_data.N != y->n_global);
  PEVSL_CHKERR(pevsl_data.n != y->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != y->n_first);

  double tms = pEVSL_Wtime();
  
  pevsl_data.Bmv->func(x->data, y->data, pevsl_data.Bmv->data);
  
  double tme = pEVSL_Wtime();
  pevsl_stat.t_mvB += tme - tms;
  pevsl_stat.n_mvB ++;
}

/**
* @brief y = B \ x
* This is the solve function for the matrix B in pevsl_Data
*/
static inline void pEVSL_SolveB(pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(!pevsl_data.Bsol);
 
  PEVSL_CHKERR(pevsl_data.N != x->n_global);
  PEVSL_CHKERR(pevsl_data.n != x->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != x->n_first);
  PEVSL_CHKERR(pevsl_data.N != y->n_global);
  PEVSL_CHKERR(pevsl_data.n != y->n_local);
  PEVSL_CHKERR(pevsl_data.nfirst != y->n_first);

  double tms = pEVSL_Wtime();
  
  pevsl_data.Bsol->func(x->data, y->data, pevsl_data.Bsol->data);
  
  double tme = pEVSL_Wtime();
  pevsl_stat.t_svB += tme - tms;
  pevsl_stat.n_svB ++;
}

/**
 * @brief check if an interval is valid 
 **/
static inline int check_intv(double *intv, FILE *fstats) {
  /* intv[4]: ( intv[0], intv[1] ) is the inteval of interest
   *          ( intv[2], intv[3] ) is the spectrum bounds
   * return   0: ok
   *        < 0: interval is invalid
   */
  double a=intv[0], b=intv[1], lmin=intv[2], lmax=intv[3];
  if (a >= b) {
    fprintf(fstats, " error: invalid interval (%e, %e)\n", a, b);
    return -1;
  }
  
  if (a >= lmax || b <= lmin) {
    fprintf(fstats, " error: interval (%e, %e) is outside (%e %e) \n", 
            a, b, lmin, lmax);
    return -2;
  } 
  
  return 0;
}

#endif

