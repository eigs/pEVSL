/*
  This file contains function prototypes and constant definitions for EVSL
*/

#ifndef PEVSL_H
#define PEVSL_H

#include "pevsl_struct.h"
#include "pevsl_def.h"

/* chebiter.c */
int pEVSL_SetupChebIterMatB(int deg, int lanm, int msteps, double tol, 
                            MPI_Comm comm, BSolDataChebiter *data);
void pEVSL_ChebIterSolMatB(double *db, double *dx, void *data, MPI_Comm comm);

/* chebpol.c */
void pEVSL_SetPolDef(pevsl_Polparams *pol);
int  pEVSL_FindPol(double *intv, pevsl_Polparams *pol);
void pEVSL_FreePol(pevsl_Polparams *pol);

/* lantrbnd.c */
int pEVSL_LanTrbounds(int lanm, int maxit, double tol, pevsl_Parvec *vinit,
                      int bndtype, double *lammin, double *lammax, MPI_Comm comm, FILE *fstats);

/* pevsl.c */
int pEVSL_Start();
int pEVSL_Finish();
int pEVSL_SetProbSizes(int N, int n, int nfirst);
int pEVSL_SetAParcsr(pevsl_Parcsr *A);
int pEVSL_SetBParcsr(pevsl_Parcsr *B);
int pEVSL_SetAMatvec(MVFunc func, void *data);
int pEVSL_SetBMatvec(MVFunc func, void *data);
int pEVSL_SetBSol(SVFunc func, void *data);
int pEVSL_SetStdEig();
int pEVSL_SetGenEig();

/* parcsr.c */
int  pEVSL_ParcsrCreate(int nrow, int ncol, int *row_starts, int *col_starts, 
                        pevsl_Parcsr *A, MPI_Comm comm);
int  pEVSL_ParcsrSetup(pevsl_Csr *Ai, pevsl_Parcsr *A);
void pEVSL_ParcsrFree(pevsl_Parcsr *A);
int  pEVSL_ParcsrGetLocalMat(pevsl_Parcsr *A, int cooidx, pevsl_Coo *coo, 
                             pevsl_Csr *csr, char stype);
int  pEVSL_ParcsrNnz(pevsl_Parcsr *A);
int  pEVSL_ParcsrLocalNnz(pevsl_Parcsr *A);

/* parcsrmv.c */
void pEVSL_ParcsrMatvec(pevsl_Parcsr *A, pevsl_Parvec *x, pevsl_Parvec *y);

/* parvec.c */
void pEVSL_ParvecCreate(int nglobal, int nlocal, int nfirst, MPI_Comm comm, pevsl_Parvec *x);
void pEVSL_ParvecCreateShell(int nglobal, int nlocal, int nfirst, MPI_Comm comm, pevsl_Parvec *x, double *data);
void pEVSL_ParvecDupl(pevsl_Parvec *x, pevsl_Parvec *y);
void pEVSL_ParvecFree(pevsl_Parvec *x);
void pEVSL_ParvecRand(pevsl_Parvec *x);
void pEVSL_ParvecDot(pevsl_Parvec *x, pevsl_Parvec *y, double *t);
void pEVSL_ParvecNrm2(pevsl_Parvec *x, double *t);
void pEVSL_ParvecCopy(pevsl_Parvec *x, pevsl_Parvec *y);
void pEVSL_ParvecSum(pevsl_Parvec *x, double *t);
void pEVSL_ParvecScal(pevsl_Parvec *x, double t);
/* void pEVSL_ParvecAddScalar(pevsl_Parvec *x, double t); */
void pEVSL_ParvecSetScalar(pevsl_Parvec *x, double t);
void pEVSL_ParvecSetZero(pevsl_Parvec *x);
void pEVSL_ParvecAxpy(double a, pevsl_Parvec *x, pevsl_Parvec *y);
int  pEVSL_ParvecSameSize(pevsl_Parvec *x, pevsl_Parvec *y);
int pEVSL_ParvecWrite(pevsl_Parvec *x, const char *fn);

/* spmat.c */
void pEVSL_CsrResize(int nrow, int ncol, int nnz, pevsl_Csr *csr);
void pEVSL_FreeCsr(pevsl_Csr *csr);
void pEVSL_CooResize(int nrow, int ncol, int nnz, pevsl_Coo *coo);
void pEVSL_FreeCoo(pevsl_Coo *coo);
int  pEVSL_CooToCsr(int cooidx, pevsl_Coo *coo, pevsl_Csr *csr);
int  pEVSL_CsrToCoo(pevsl_Csr *csr, int cooidx, pevsl_Coo *coo);
int  pEVSL_MatvecGen(double alp, pevsl_Csr *A, double *x, double bet, double *y);
int  pEVSL_Matvec(pevsl_Csr *A, double *x, double *y);
void pEVSL_SortRow(pevsl_Csr *A);

/* utils.c */
void pEVSL_SortInt(int *x, int n);
void pEVSL_SortDouble(int n, double *v, int *ind);
void pEVSL_LinSpace(double a, double b, int num, double *arr);
void pEVSL_Part1d(int len, int pnum, int *idx, int *j1, int *j2, int job);
int  pEVSL_BinarySearch(int *x, int n, int key);
void pEVSL_Vecset(int n, double t, double *v);

/* vector.c */
void linspace(double a, double b, int num, double *arr);
void vecset(int n, double t, double *v);
void sort_double(int n, double *v, int *ind);

/* stats.c */
void pEVSL_StatsReset();
void pEVSL_StatsPrint(FILE *fstats, MPI_Comm comm);

/* cheblanNr.c */
int pEVSL_ChebLanNr(double *intv, int maxit, double tol, pevsl_Parvec *vinit, 
                    pevsl_Polparams *pol, int *nevOut, double **lamo, 
                    pevsl_Parvec **Wo, double **reso, MPI_Comm comm, FILE *fstats);
#endif

