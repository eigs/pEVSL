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

int pEVSL_SetAParcsr(pevsl_Parcsr *A);

int pEVSL_SetBParcsr(pevsl_Parcsr *B);

int pEVSL_SetAMatvec(int N, int n, MVFunc func, void *data);

int pEVSL_SetBMatvec(int N, int n, MVFunc func, void *data);

int pEVSL_SetBSol(SolFuncR func, void *data);

int pEVSL_SetStdEig();

int pEVSL_SetGenEig();

#endif

