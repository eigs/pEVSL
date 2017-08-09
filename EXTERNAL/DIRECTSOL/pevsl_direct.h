#ifndef PEVSL_DIRECT_H
#define PEVSL_DIRECT_H

/**
 * @file evsl_direct.h
 * @brief Definitions used for direct solver interface
 * 
 * Note that this file is meant to be the header file
 * of both pevsl_mumps.c, pevsl_pardiso.c,
 * If more direct solver options will be added later,
 * this should serve them as well.
 */

/* functions for B solve */
int  SetupBSolDirect(pevsl_Parcsr *B, void **data);
void BSolDirect(double *b, double *x, void *data);
void LTSolDirect(double *b, double *x, void *data);
void DSolDirect(double *b, double *x, void *data);
void FreeBSolDirectData(void *data);

#endif
