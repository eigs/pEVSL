/*
  This file contains function prototypes and constant definitions for EVSL
*/

#ifndef PEVSL_H
#define PEVSL_H

#include "pevsl_struct.h"
#include "pevsl_def.h"

/* chebpol.c */
void pEVSL_SetPolDef(pevsl_Polparams *pol);
int  pEVSL_FindPol(double *intv, pevsl_Polparams *pol);
void pEVSL_FreePol(pevsl_Polparams *pol);

/* kpmdos.c */
int pEVSL_Kpmdos(pevsl_Data *pevsl, int Mdeg, int damping, int nvec, double *intv,
                 int ngroups, int groupid, MPI_Comm gl_comm, double *mu, double *ecnt);

/* lantrbnd.c */
int pEVSL_LanTrbounds(pevsl_Data *pevsl, int lanm, int maxit, double tol, 
                      pevsl_Parvec *vinit, int bndtype, double *lammin, 
                      double *lammax, FILE *fstats);

/* pevsl.c */
int pEVSL_Start       (MPI_Comm comm, pevsl_Data **data);
int pEVSL_Finish      (pevsl_Data *pevsl_data);
int pEVSL_SetProbSizes(pevsl_Data *pevsl_data, int N, int n, int nfirst);
int pEVSL_SetAParcsr  (pevsl_Data *pevsl_data, pevsl_Parcsr *A);
int pEVSL_SetBParcsr  (pevsl_Data *pevsl_data, pevsl_Parcsr *B);
int pEVSL_SetAMatvec  (pevsl_Data *pevsl_data, MVFunc func, void *data);
int pEVSL_SetBMatvec  (pevsl_Data *pevsl_data, MVFunc func, void *data);
int pEVSL_SetBSol     (pevsl_Data *pevsl_data, SVFunc func, void *data);
int pEVSL_SetLTSol    (pevsl_Data *pevsl_data, SVFunc func, void *data);
int pEVSL_SetStdEig   (pevsl_Data *pevsl_data);
int pEVSL_SetGenEig   (pevsl_Data *pevsl_data);

/* parcsr.c */
int  pEVSL_ParcsrCreate(int nrow, int ncol, int *row_starts, int *col_starts, 
                        pevsl_Parcsr *A, MPI_Comm comm);
int  pEVSL_ParcsrSetup(pevsl_Csr *Ai, pevsl_Parcsr *A);
void pEVSL_ParcsrFree(pevsl_Parcsr *A);
int  pEVSL_ParcsrGetLocalMat(pevsl_Parcsr *A, int idx, pevsl_Coo *coo, 
                             pevsl_Csr *csr, char stype);
int  pEVSL_ParcsrNnz(pevsl_Parcsr *A);
int  pEVSL_ParcsrLocalNnz(pevsl_Parcsr *A);

/* parcsrmv.c */
void pEVSL_ParcsrMatvec(pevsl_Parcsr *A, pevsl_Parvec *x, pevsl_Parvec *y);

/* parvec.c */
void pEVSL_ParvecCreate(int nglobal, int nlocal, int nfirst, MPI_Comm comm, pevsl_Parvec *x);
void pEVSL_ParvecCreateShell(int nglobal, int nlocal, int nfirst, MPI_Comm comm,
                             pevsl_Parvec *x, double *data);
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
int  pEVSL_ParvecWrite(pevsl_Parvec *x, const char *fn);

/* parvecs.c */
void pEVSL_ParvecsCreateShell(int nglobal, int nvecs, int ld, int nlocal, int nfirst, 
                              MPI_Comm comm, pevsl_Parvecs *x, double *data);

void pEVSL_ParvecsCreate(int nglobal, int nvecs, int ld, int nlocal, int nfirst,
                         MPI_Comm comm, pevsl_Parvecs *x);

void pEVSL_ParvecsDuplParvec(pevsl_Parvec *x, int nvecs, int ld, pevsl_Parvecs *y);

void pEVSL_ParvecsFree(pevsl_Parvecs *x);

void pEVSL_ParvecsGetParvecShell(pevsl_Parvecs *X, int i, pevsl_Parvec *x);

void pEVSL_ParvecsGemv(double alp, pevsl_Parvecs *A, int nvecs, double *x, 
                       double bet, pevsl_Parvec *y);

void pEVSL_ParvecsGemtvWithWspace(double alp, pevsl_Parvecs *A, int nvecs, pevsl_Parvec *x, 
                                  double bet, double *y, double *w);

void pEVSL_ParvecsGemtv(double alp, pevsl_Parvecs *A, int nvecs, pevsl_Parvec *x, 
                        double bet, double *y);

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
double pEVSL_Wtime();

/* vector.c */
void linspace(double a, double b, int num, double *arr);
void vecset(int n, double t, double *v);
void sort_double(int n, double *v, int *ind);

/* spslicer.c */
int pEVSL_Spslicer(double *sli, double *mu, int Mdeg, double *intv, int n_int, int npts);

/* stats.c */
void pEVSL_StatsReset(pevsl_Data *pevsl);
void pEVSL_StatsPrint(pevsl_Data *pevsl, FILE *fstats);

/* cheblanNr.c */
int pEVSL_ChebLanNr(pevsl_Data *pevsl, double *intv, int maxit, double tol, pevsl_Parvec *vinit, 
                    pevsl_Polparams *pol, int *nevOut, double **lamo, pevsl_Parvecs **Wo, 
                    double **reso, FILE *fstats);

#endif

