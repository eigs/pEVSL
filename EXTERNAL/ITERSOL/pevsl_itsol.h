#ifndef PEVSL_ITSOL_H
#define PEVSL_ITSOL_H

/**
 * @file pevsl_itsol.h
 * @brief Definitions used for iterative solver interface
 * 
 * Note that this file is meant to be the header file
 * of chebiter.c,
 * If more iterative solver options will be added later,
 * this should serve them as well.
 */

/* functions for Chebyshev iterations */
int pEVSL_ChebIterSetup(double lmin, double lmax, int deg, pevsl_Parcsr *A, 
                        void **data);

void pEVSL_ChebIterSolv1(double *db, double *dx, void *data);

void pEVSL_ChebIterSolv2(double *db, double *dx, void *data);

void pEVSL_ChebIterFree(void *vdata);

double* pEVSL_ChebIterGetRes(void *data);

void pEVSL_ChebIterStatsPrint(void *data, FILE *fstats);

#endif
