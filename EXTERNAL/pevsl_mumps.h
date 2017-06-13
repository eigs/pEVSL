#ifndef PEVSL_MUMPS_H
#define PEVSL_MUMPS_H

#include "dmumps_c.h"

typedef struct _BSolDataMumps {
  DMUMPS_STRUC_C solver;
  double *rhs_global;
  int *ncols;
  int *icols;
} BSolDataMumps;

/* B-solve routines with MUMPS */
void BSolMumps(double *b, double *x, void *data);

int SetupBSolMumps(pevsl_Parcsr *B, BSolDataMumps *data);

int FreeBSolMumps(BSolDataMumps *data);
#endif
