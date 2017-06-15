/*
  This file contains function prototypes and constant definitions for EVSL
*/

#ifndef PEVSL_H
#define PEVSL_H

#include "pevsl_struct.h"
#include "pevsl_def.h"

#include "pevsl_protos.h"

int pEVSL_Start();

int pEVSL_Finish();

int pEVSL_SetProbSizes(int N, int n, int nfirst);

int pEVSL_SetAParcsr(pevsl_Parcsr *A);

int pEVSL_SetBParcsr(pevsl_Parcsr *B);

int pEVSL_SetAMatvec(MVFunc func, void *data);

int pEVSL_SetBMatvec(MVFunc func, void *data);

int pEVSL_SetBSol(SolFuncR func, void *data);

int pEVSL_SetStdEig();

int pEVSL_SetGenEig();

void pEVSL_StatsReset();

void pEVSL_StatsPrint(FILE *fstats, MPI_Comm comm);

int pEVSL_ChebLanNr(double *intv, int maxit, double tol, pevsl_Parvec *vinit, 
                    pevsl_Polparams *pol, int *nevOut, double **lamo, 
                    pevsl_Parvec **Wo, double **reso, MPI_Comm comm, FILE *fstats);
#endif

