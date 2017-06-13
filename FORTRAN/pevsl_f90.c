#include "pevsl.h"

/** @brief Fortran interface for pEVSL_Start */
void PEVSL_FORT(pevsl_start)() {
  pEVSL_Start();
}

/** @brief Fortran interface for pEVSL_Finish */
void PEVSL_FORT(pevsl_finish)() {
  pEVSL_Finish();
}

/** @brief Fortran interface for pEVSL_SetAMatvec
 * @param[in] N : global size of A 
 * @param[in] n : local size of A
 * @param[in] func : function pointer 
 * @param[in] data : associated data
 */
void PEVSL_FORT(pevsl_setamv)(int *N, int *n, void *func, void *data) {
  pEVSL_SetAMatvec(*N, *n, (MVFunc) func, data);
}

/*
void PEVSL_FORT(pevsl_parveccreate)(int *N, int *n, MPI_Fint *Fcomm, uintptr_t *xf90) {
  pevsl_Parvec *x;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  PEVSL_MALLOC(x, 1, pevsl_Parvec);
  pEVSL_ParvecCreate(*N, *n, 0, comm, x);
  
  *xf90 = (uintptr_t) x;
}

void PEVSL_FORT(pevsl_parvecsetvals)(uintptr_t *xf90, double *vals) {
  int i, n;
  pevsl_Parvec *x = (pevsl_Parvec *) xf90;
  n = x->n_local;
  for (i=0; i<n; i++) {
    x->data[i] = vals[i];
  }
}

void PEVSL_FORT(pevsl_parvecgetvals)(uintptr_t *xf90, double *vals) {
  int i, n;
  pevsl_Parvec *x = (pevsl_Parvec *) xf90;
  n = x->n_local;
  for (i=0; i<n; i++) {
    vals[i] = x->data[i];
  }
}

void PEVSL_FORT(pevsl_parvecfree)(uintptr_t *xf90) {
  pevsl_Parvec *x = (pevsl_Parvec *) xf90;
  pEVSL_ParvecFree(x);
  PEVSL_FREE(x);
}
*/

void PEVSL_FORT(pevsl_amv)(double *x, double *y) {
  pevsl_data.Amv->func(x, y, pevsl_data.Amv->data);
}

/** @brief Fortran interface for evsl_lanbounds 
 * @param[in] nstpes: number of steps
 * @param[out] lmin: lower bound
 * @param[out] lmax: upper bound
 * */
void PEVSL_FORT(pevsl_lanbounds)(int *nsteps, double *lmin, double *lmax, MPI_Fint *Fcomm) {
  int N, n;
  pevsl_Parvec vinit;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, 0, comm, &vinit);
  pEVSL_ParvecRand(&vinit);
  /*------------------- Lanczos Bounds */
  pEVSL_LanTrbounds(50, *nsteps, 1e-10, &vinit, 1, lmin, lmax, comm, NULL);
    
  pEVSL_ParvecFree(&vinit);
}

/** @brief Fortran interface for find_pol 
 * @param[out] polf90 : pointer of pol
 * @warning: The pointer will be cast to uintptr_t
 * 
 * uintptr_t: Integer type capable of holding a value converted from 
 * a void pointer and then be converted back to that type with a value 
 * that compares equal to the original pointer
 */
void PEVSL_FORT(pevsl_findpol)(double *xintv, double *thresh_int, 
                               double *thresh_ext, uintptr_t *polf90) {
  pevsl_Polparams *pol;
  PEVSL_MALLOC(pol, 1, pevsl_Polparams);
  pEVSL_SetPolDef(pol);
  pol->damping = 2;
  pol->thresh_int = *thresh_int;
  pol->thresh_ext = *thresh_ext;
  pol->max_deg  = 500;
  pEVSL_FindPol(xintv, pol);       
  
  fprintf(stdout, " polynomial deg %d, bar %e gam %e\n",
          pol->deg,pol->bar, pol->gam);

  *polf90 = (uintptr_t) pol;
}

/** @brief Fortran interface for free_pol */
void PEVSL_FORT(pevsl_freepol)(uintptr_t *polf90) {
  /* cast pointer */
  pevsl_Polparams *pol = (pevsl_Polparams *) (*polf90);
  pEVSL_FreePol(pol);
  PEVSL_FREE(pol);
}

/** @brief Fortran interface for ChebLanNr
 *  the results will be saved in the internal variables
 */
void PEVSL_FORT(pevsl_cheblannr)(double *xintv, int *max_its, double *tol, 
                                 MPI_Fint *Fcomm, uintptr_t *polf90) {
  int N, n, nev2, ierr;
  double *lam, *res;
  pevsl_Parvec *Y;
  FILE *fstats = stdout;
  pevsl_Parvec vinit;
 
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, 0, comm, &vinit);
  pEVSL_ParvecRand(&vinit);
  /* cast pointer of pol*/
  pevsl_Polparams *pol = (pevsl_Polparams *) (*polf90);
  /* call ChebLanNr */ 
  ierr = pEVSL_ChebLanNr(xintv, *max_its, *tol, &vinit, pol, &nev2, &lam, 
                         &Y, &res, fstats);

  if (ierr) {
    printf("ChebLanNr error %d\n", ierr);
  }

  pEVSL_ParvecFree(&vinit);
  if (res) {
    free(res);
  }
  /* save pointers to the global variables */
  //evsl_nev_computed = nev2;
  //evsl_n = n;
  //evsl_eigval_computed = lam;
  //evsl_eigvec_computed = Y;
}

#if 0
void PEVSL_FORT(pevsl_test)(MPI_Fint *Fcomm) {
  int N, n;
  pevsl_Parvec x, y;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, 0, comm, &x);
  pEVSL_ParvecRand(&x);
  //pEVSL_ParvecSetScalar(&x, 3.14);
  pEVSL_ParvecDupl(&x, &y);

  pEVSL_MatvecA(&x, &y);
  double t;  
  pEVSL_ParvecNrm2(&y, &t);
  
  printf("t = %.15e\n", t);
  
  pEVSL_ParvecFree(&x);
  pEVSL_ParvecFree(&y);
}


void PEVSL_FORT(pevsl_testc)(MPI_Comm comm) {
  int N, n;
  double t;  

  pevsl_Parvec x;
  N = pevsl_data.N;
  n = pevsl_data.n;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, 0, comm, &x);
  pEVSL_ParvecRand(&x);
  pEVSL_ParvecNrm2(&x, &t);

  //pEVSL_ParvecSetScalar(&x, 3.14);
  //pEVSL_ParvecDupl(&x, &y);

  //pEVSL_MatvecA(&x, &y);
  //pEVSL_ParvecNrm2(&y, &t);
  
  printf("N = %d, n = %d, t = %.15e\n", N, n, t);
  
  pEVSL_ParvecFree(&x);
  //pEVSL_ParvecFree(&y);
}

#endif

