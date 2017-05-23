/*
  This file contains function prototypes and constant definitions for EVSL
*/

#ifndef PEVSL_PROTOS_H
#define PEVSL_PROTOS_H

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <float.h>

#include "pevsl_struct.h"
#include "pevsl_def.h"

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


/* cheblanNr.c */
int pEVSL_ChebLanNr(double *intv, int maxit, double tol, pevsl_Parvec *vinit, 
                    pevsl_Polparams *pol, int *nevOut, double **lamo, pevsl_Parvec **Wo, 
                    double **reso, pevsl_Comm *comm, FILE *fstats);

/* comm.c */
int pEVSL_CommCreate(pevsl_Comm *comm, MPI_Comm comm_global, int ngroups);

void pEVSL_CommFree(pevsl_Comm *comm);

/* chebpol.c */
void pEVSL_SetPolDef(pevsl_Polparams *pol);

int pEVSL_FindPol(double *intv, pevsl_Polparams *pol);

void pEVSL_FreePol(pevsl_Polparams *pol);

int pEVSL_ChebAv(pevsl_Polparams *pol, pevsl_Parvec *v, pevsl_Parvec *y, pevsl_Parvec *w);

/* miscla.c */
int SymmTridEig(double *eigVal, double *eigVec, int n, const double *diag, const double *sdiag);

int SymmTridEigS(double *eigVal, double *eigVec, int n, double vl, double vu,
                 int *nevO, const double *diag, const double *sdiag);
                 
void SymEigenSolver(int n, double *A, int lda, double *Q, int ldq, double *lam);

void MGS_DGKS(int k, int i_max, pevsl_Parvec *Q, pevsl_Parvec *v, double *nrmv);

void MGS_DGKS2(int k, int i_max, pevsl_Parvec *Z, pevsl_Parvec *Q, pevsl_Parvec *v);            

/* parcsr.c */
int pEVSL_ParcsrCreate(int nrow, int ncol, int *row_starts, int *col_starts, pevsl_Parcsr *A, MPI_Comm comm);

int pEVSL_ParcsrSetup(pevsl_Csr *Ai, pevsl_Parcsr *A);

void pEVSL_ParcsrFree(pevsl_Parcsr *A);

/* parcsrmv.c */
void pEVSL_ParcsrMatvec(pevsl_Parvec *x, pevsl_Parvec *y, void *data);

/* parvec.c */
void pEVSL_ParvecCreate(int nglobal, int nlocal, int nfirst, MPI_Comm comm, pevsl_Parvec *x);

void pEVSL_ParvecDupl(pevsl_Parvec *x, pevsl_Parvec *y);

void pEVSL_ParvecFree(pevsl_Parvec *x);

void pEVSL_ParvecRand(pevsl_Parvec *x);

void pEVSL_ParvecDot(pevsl_Parvec *x, pevsl_Parvec *y, double *t);

void pEVSL_ParvecNrm2(pevsl_Parvec *x, double *t);

void pEVSL_ParvecCopy(pevsl_Parvec *x, pevsl_Parvec *y);

void pEVSL_ParvecSum(pevsl_Parvec *x, double *t);

void pEVSL_ParvecScal(pevsl_Parvec *x, double t);

void pEVSL_ParvecAddScalar(pevsl_Parvec *x, double t);

void pEVSL_ParvecSetScalar(pevsl_Parvec *x, double t);

void pEVSL_ParvecAxpy(double a, pevsl_Parvec *x, pevsl_Parvec *y);

int pEVSL_ParvecSameSize(pevsl_Parvec *x, pevsl_Parvec *y);


/* spmat.c */
void pEVSL_CsrResize(int nrow, int ncol, int nnz, pevsl_Csr *csr);

void pEVSL_FreeCsr(pevsl_Csr *csr);

void pEVSL_FreeCoo(pevsl_Coo *coo);

int pEVSL_CooToCsr(int cooidx, pevsl_Coo *coo, pevsl_Csr *csr);

int pEVSL_MatvecGen(double alp, pevsl_Csr *A, double *x, double bet, double *y);

int pEVSL_Matvec(pevsl_Csr *A, double *x, double *y);


/* utils.c */
void pEVSL_SortInt(int *x, int n);

void pEVSL_SortDouble(int n, double *v, int *ind);

void pEVSL_LinSpace(double a, double b, int num, double *arr);

void pEVSL_Part1d(int len, int pnum, int *idx, int *j1, int *j2, int job);

int pEVSL_BinarySearch(int *x, int n, int key);

void pEVSL_Vecset(int n, double t, double *v);


/* vector.c */
void linspace(double a, double b, int num, double *arr);

void vecset(int n, double t, double *v);

void sort_double(int n, double *v, int *ind);



/*------------------- inline functions */

/** 
 * @brief printf, only for group_rank == 0
 * 
 * */
static inline void pEVSL_Printf0(pevsl_Comm *comm, FILE *fp, const char *format, ...) {
  if (comm->group_rank) {
    return;
  }
  va_list args;
  va_start(args, format);
  vfprintf(fp, format, args);
  va_end(args);
}


/** 
 * @brief printf, only for global_rank == 0
 * 
 * */
static inline void pEVSL_Printf00(pevsl_Comm *comm, FILE *fp, const char *format, ...) {
  if (comm->global_rank) {
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
  pevsl_data.Amv->func(x, y, pevsl_data.Amv->data);
}


/** 
 * @brief Perform matrix-vector product y = B * x
 * 
 * */

static inline void pEVSL_MatvecB(pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(!pevsl_data.Bmv);
  pevsl_data.Bmv->func(x, y, pevsl_data.Bmv->data);
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

