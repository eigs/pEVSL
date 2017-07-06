#include "pevsl_int.h"

/** global variables that hold results from EVSL
 * evsl_copy_result_f90 will copy results from these vars and reset them
 * */
int pevsl_nev_computed=0, pevsl_n=0;
double *pevsl_eigval_computed=NULL;
pevsl_Parvec *pevsl_eigvec_computed=NULL;

/** @brief Fortran interface for pEVSL_Start */
void PEVSL_FORT(pevsl_start)() {
  pEVSL_Start();
}

/** @brief Fortran interface for pEVSL_Finish */
void PEVSL_FORT(pevsl_finish)() {
  pEVSL_Finish();
}

/** @brief Fortran interface for pEVSL_SetProbSizes
 * @param[in] N : global size of A
 * @param[in] n : local size of A
 * @param[in] nfirst : nfirst
 * @warning set nfirst < 0, if do not want to specify
 */
void PEVSL_FORT(pevsl_setprobsizes)(int *N, int *n, int *nfirst) {
  pEVSL_SetProbSizes(*N, *n, *nfirst);
}

/** @brief Fortran interface for pEVSL_SetAMatvec
 * @param[in] func : function pointer 
 * @param[in] data : associated data
 */
void PEVSL_FORT(pevsl_setamv)(void *func, void *data) {
  pEVSL_SetAMatvec((MVFunc) func, data);
}

/** @brief Fortran interface for pEVSL_SetAMatvec
 * @param[in] func : function pointer 
 * @param[in] data : associated data
 */
void PEVSL_FORT(pevsl_setbmv)(void *func, void *data) {
  pEVSL_SetBMatvec((MVFunc) func, data);
}

/** @brief Fortran interface for SetBsol */
void PEVSL_FORT(pevsl_setbsol)(void *func, void *data) {
  pEVSL_SetBSol((SVFunc) func, data);
}


/** @brief Fortran interface for SetGenEig */
void PEVSL_FORT(pevsl_set_geneig)() {
  pEVSL_SetGenEig();
}

void PEVSL_FORT(pevsl_parcsrcreate)(int *nrow, int *ncol, int *row_starts, int *col_starts,
                int *ia, int *ja, double *aa,
                MPI_Fint *Fcomm, uintptr_t *matf90) {
  pevsl_Parcsr *mat;
  pevsl_Csr matloc;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);

  PEVSL_MALLOC(mat, 1, pevsl_Parcsr);
  pEVSL_ParcsrCreate(*nrow, *ncol, row_starts, col_starts, mat, comm);

  /* matloc is the local CSR wrapper */
  matloc.nrows = mat->nrow_local;
  matloc.ncols = mat->ncol_global;
  matloc.ia = ia;
  matloc.ja = ja;
  matloc.a  = aa;

  /* setup parcsr with matloc */
  pEVSL_ParcsrSetup(&matloc, mat);

  *matf90 = (uintptr_t) mat;
}

/* perform matvec with Parcsr matrix */
void PEVSL_FORT(pevsl_parcsrmatvec)(double *x, double *y, uintptr_t *matf90) {
  /* cast pointer */
  pevsl_Parcsr *mat = (pevsl_Parcsr *) (*matf90);
  pEVSL_ParcsrMatvec0(x, y, (void *) mat);
}

void PEVSL_FORT(pevsl_seta_parcsr)(uintptr_t *Af90) {
  /* cast pointer */
  pevsl_Parcsr *A = (pevsl_Parcsr *) (*Af90);

  pEVSL_SetAParcsr(A);
}

void PEVSL_FORT(pevsl_setb_parcsr)(uintptr_t *Bf90) {
  /* cast pointer */
  pevsl_Parcsr *B = (pevsl_Parcsr *) (*Bf90);

  pEVSL_SetBParcsr(B);
}

void PEVSL_FORT(pevsl_parcsr_free)(uintptr_t *Af90) {
  /* cast pointer */
  pevsl_Parcsr *A = (pevsl_Parcsr *) (*Af90);

  pEVSL_ParcsrFree(A);
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

void PEVSL_FORT(pevsl_amv)(double *x, double *y) {
  pevsl_data.Amv->func(x, y, pevsl_data.Amv->data);
}
*/

void PEVSL_FORT(pevsl_bsv)(double *b, double *x) {
  pevsl_data.Bsol->func(b, x, pevsl_data.Bsol->data);
}

#if 0
void PEVSL_FORT(pevsl_test)(MPI_Fint *Fcomm) {
  int N, n, nfirst;
  double nrmv, nrmy, nrmz;
  pevsl_Parvec vinit, y, z;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  nfirst = pevsl_data.nfirst;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, nfirst, comm, &vinit);
  //pEVSL_ParvecRand(&vinit);
  //pEVSL_ParvecSin(&vinit);
  pEVSL_ParvecSetScalar(&vinit, 1.0);
  pEVSL_ParvecDupl(&vinit, &y);
  pEVSL_ParvecDupl(&vinit, &z);

  pEVSL_ParvecNrm2(&vinit, &nrmv);
  printf("norm v %.15e\n", nrmv);

  pEVSL_MatvecA(&vinit, &y);
  pEVSL_ParvecNrm2(&y, &nrmy);
  printf("norm y %.15e\n", nrmy);

  pEVSL_MatvecB(&vinit, &z);
  pEVSL_ParvecNrm2(&z, &nrmz);
  printf("norm z %.15e\n", nrmz);

  pEVSL_SolveB(&vinit, &y);
  pEVSL_ParvecNrm2(&y, &nrmy);
  printf("norm y2 %.15e\n", nrmy);

  //pEVSL_ParvecAxpy(-1.0, &vinit, &y);
  //pEVSL_ParvecNrm2(&y, &nrmy);
  //printf("norm y3 %.15e\n", nrmy);
}
#endif

/** @brief Fortran interface for evsl_lanbounds 
 * @param[in] nstpes: number of steps
 * @param[out] lmin: lower bound
 * @param[out] lmax: upper bound
 * */
void PEVSL_FORT(pevsl_lanbounds)(int *mlan, int *nsteps, double *lmin, 
                                 double *lmax, MPI_Fint *Fcomm) {
  int N, n, nfirst;
  pevsl_Parvec vinit;
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  nfirst = pevsl_data.nfirst;
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, nfirst, comm, &vinit);
  pEVSL_ParvecRand(&vinit);
  /*------------------- Lanczos Bounds */
  pEVSL_LanTrbounds(*mlan, *nsteps, 1e-8, &vinit, 1, lmin, lmax, comm, NULL);
    
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
  int N, n, nfirst, nev2, ierr;
  double *lam, *res;
  pevsl_Parvec *Y;
  FILE *fstats = stdout;
  pevsl_Parvec vinit;
 
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  nfirst = pevsl_data.nfirst;
  /*-------------------- zero out stats */
  pEVSL_StatsReset();
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(N, n, nfirst, comm, &vinit);
  pEVSL_ParvecRand(&vinit);
  /* cast pointer of pol*/
  pevsl_Polparams *pol = (pevsl_Polparams *) (*polf90);
  /* call ChebLanNr */ 
  ierr = pEVSL_ChebLanNr(xintv, *max_its, *tol, &vinit, pol, &nev2, &lam, 
                         &Y, &res, comm, fstats);

  if (ierr) {
    printf("ChebLanNr error %d\n", ierr);
  }
  pEVSL_ParvecFree(&vinit);
  /*--------------------- print stats */
  pEVSL_StatsPrint(fstats, comm);

  if (res) {
    free(res);
  }
  /* save pointers to the global variables */
  pevsl_nev_computed = nev2;
  pevsl_n = n;
  pevsl_eigval_computed = lam;
  pevsl_eigvec_computed = Y;
}

/** @brief Get the number of last computed eigenvalues 
 */
void PEVSL_FORT(pevsl_get_nev)(int *nev) {
  *nev = pevsl_nev_computed;
}

/** @brief copy the computed eigenvalues and vectors
 * @warning: after this call the internal saved results will be freed
 */
void PEVSL_FORT(pevsl_copy_result)(double *val, double *vec) {
  int i;
  /* copy eigenvalues */
  memcpy(val, pevsl_eigval_computed, pevsl_nev_computed*sizeof(double));
  /* copy eigenvectors */
  for (i=0; i<pevsl_nev_computed; i++) {
    memcpy(vec+i*pevsl_n, pevsl_eigvec_computed[i].data, pevsl_n*sizeof(double));
  }
  /* reset global variables */
  pevsl_nev_computed = 0;
  pevsl_n = 0;
  free(pevsl_eigval_computed);
  pevsl_eigval_computed = NULL;
  for (i=0; i<pevsl_nev_computed; i++) {
    pEVSL_ParvecFree(&pevsl_eigvec_computed[i]);
  }
  free(pevsl_eigvec_computed);
  pevsl_eigvec_computed = NULL;
}

void PEVSL_FORT(pevsl_setup_chebiter)(double *lmin, double *lmax, int *deg, 
                                      uintptr_t *parcsrf90, MPI_Fint *Fcomm, 
                                      uintptr_t *chebf90) {
  /* cast pointer of the matrix */
  pevsl_Parcsr *A = (pevsl_Parcsr *) (*parcsrf90);
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  Chebiter_Data *cheb;
  PEVSL_MALLOC(cheb, 1, Chebiter_Data);
  pEVSL_ChebIterSetup(*lmin, *lmax, *deg, A, comm, cheb);
  
  *chebf90 = (uintptr_t) cheb;
}

/** @brief Fortran interface for ChebIterSol */
void PEVSL_FORT(pevsl_chebiter)(int *type, double *b, double *x, uintptr_t *chebf90) {
  /* cast pointer */
  Chebiter_Data *cheb = (Chebiter_Data *) (*chebf90);
  if (*type == 1) {
    pEVSL_ChebIterSolv1(b, x, cheb);
  } else {
    pEVSL_ChebIterSolv2(b, x, cheb);
  }
}

/** @brief Fortran interface for ChebIterFree */
void PEVSL_FORT(pevsl_free_chebiterb)(uintptr_t *chebf90) {
  /* cast pointer */
  Chebiter_Data *cheb = (Chebiter_Data *) (*chebf90);
  pEVSL_ChebIterFree(cheb);
}


/** @brief Fortran interface for SetBsol with ChebIterSol */
void PEVSL_FORT(pevsl_setbsol_chebiter)(int *type, void *data) {
  if (*type == 1) {
    pEVSL_SetBSol(pEVSL_ChebIterSolv1, data);
  } else {
    pEVSL_SetBSol(pEVSL_ChebIterSolv2, data);
  }
}

#if 0
void PEVSL_FORT(pevsl_testchebiterb)(uintptr_t *chebf90, MPI_Fint *Fcomm) {
  int i, N, n, nfirst;
  pevsl_Parvec y,b,x,d;
 
  ChebiterData *cheb = (ChebiterData *) (*chebf90);
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  N = pevsl_data.N;
  n = pevsl_data.n;
  nfirst = pevsl_data.nfirst;

  pEVSL_ParvecCreate(N, n, nfirst, comm, &b);
  pEVSL_ParvecDupl(&b, &y);
  pEVSL_ParvecDupl(&b, &x);
  pEVSL_ParvecDupl(&b, &d);
  
  /* y is the exact sol */
  pEVSL_ParvecRand(&y);
  /* rhs: b = B*y */
  pEVSL_MatvecB(&y, &b);

  pEVSL_SolveB(&b, &x);
  
  if (cheb->res) {
    printf("CHEB ITER RES\n");
    for (i=0; i<cheb->deg+1; i++) {
      printf("i %3d: %e\n", i, cheb->res[i]);
    }
  }

  pEVSL_MatvecB(&x, &d);
  pEVSL_ParvecAxpy(-1.0, &b, &d);
  pEVSL_ParvecAxpy(-1.0, &y, &x);
  double err_r, err_x;
  pEVSL_ParvecNrm2(&d, &err_r);
  pEVSL_ParvecNrm2(&x, &err_x);

  printf("||res|| %e, ||err|| %e\n", err_r, err_x); 
}
#endif
