#ifndef PEVSL_MUMPS_H
#define PEVSL_MUMPS_H

#include "dmumps_c.h"

typedef struct _BSolDataMumps {
  /* global and local size */
  int N, n;
  DMUMPS_STRUC_C solver;
  double *rhs_global;
  /* number of local entries and start of local entries
   * only exists on rank 0, needed by MUMPS since we need to 
   * centralize rhs that needs to do Gatherv */
  int *ncols;
  int *icols;
} BSolDataMumps;

/* B-solve routines with MUMPS */
void BSolMumps(double *b, double *x, void *data);

int SetupBSolMumps(pevsl_Parcsr *B, BSolDataMumps *data);

int FreeBSolMumps(BSolDataMumps *data);
#endif
