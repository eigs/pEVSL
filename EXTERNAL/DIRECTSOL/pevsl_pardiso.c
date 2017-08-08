#include "pevsl_int.h"
#include "pevsl_direct.h"
#include "mkl.h"
#include "mkl_cluster_sparse_solver.h"

/**
 * @file pevsl_pardiso.c
 * @brief Definitions used for Pardiso interface
 */

typedef struct _BSolDataDirect {
  /* Internal solver memory pointer pt, */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
  /* or void *pt[64] should be OK on both architectures */
  void *pt[64];
  /* Pardiso control parameters. */
  MKL_INT iparm[64];
  /* global size */
  int n;
  /* upper triangular matrix */
  pevsl_Csr U;
  /* Fortran MPI communicator */
  MPI_Fint commf;
} BSolDataDirect;

/** @brief Setup the B-sol by factorization with Pardiso
 *
 * @param B      parcsr matrix B
 * */
int SetupBSolDirect(pevsl_Parcsr *B, void **data) {

  double tms = pEVSL_Wtime();

  int i, rank, size;
  BSolDataDirect *Bsol_data;
  PEVSL_MALLOC(Bsol_data, 1, BSolDataDirect);

  MPI_Comm comm = B->comm;
  MPI_Fint commf = MPI_Comm_c2f(comm);
  Bsol_data->commf = commf;
  
  /*-------------------- MPI rank and size in comm */
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  MKL_INT mtype = 2; /* Real SPD matrix */
  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  /* Controls the execution of the solver. Usually it is a two- or 
   * three-digit integer. The first digit indicates the starting phase 
   * of execution and the second digit indicates the ending phase. */
  MKL_INT phase;
  /* Message level information. If msglvl = 0 then pardiso generates no 
   * output, if msglvl = 1 the solver prints statistical information 
   * to the screen */
  MKL_INT msglvl = 0;
  /* Initialize error flag */
  MKL_INT error = 0;
  /* Double dummy */
  double ddum;
  /* Integer dummy. */
  MKL_INT idum;
  /* Number of right hand sides. */
  MKL_INT nrhs = 1;

  for ( i = 0; i < 64; i++ ) {
    Bsol_data->iparm[i] = 0;
  }
  for ( i = 0; i < 64; i++ ) {
    Bsol_data->pt[i] = 0;
  }

  /* ----------------------------------------------------------- */
  /* Setup Cluster Sparse Solver control parameters.             */
  /* ----------------------------------------------------------- */
  Bsol_data->iparm[ 0] =  1; /* Solver default parameters overriden with provided by iparm */
  Bsol_data->iparm[ 1] =  2; /* Use METIS for fill-in reordering */
  Bsol_data->iparm[ 5] =  0; /* Write solution into x */
  Bsol_data->iparm[ 7] =  0; /* Max number of iterative refinement steps */
  Bsol_data->iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
  Bsol_data->iparm[10] =  0; /* Don't use nonsymmetric permutation and scaling MPS */
  Bsol_data->iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
  Bsol_data->iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  Bsol_data->iparm[18] = -1; /* Output: Mflops for LU factorization */
  Bsol_data->iparm[26] =  1; /* Check input data for correctness */
  /* Cluster Sparse Solver use C-style indexing for ia and ja arrays */
  Bsol_data->iparm[34] =  1;
  /* If iparm[39]=2, the matrix is provided in distributed assembled matrix input
     format. In this case, each MPI process stores only a part (or domain) of the matrix A 
     data. The bounds of the domain should be set via iparm(41) and iparm(42). Solution    
     vector is distributed between process in same manner with rhs. */
  /* Input: matrix/rhs/solution are distributed between MPI processes  */
  Bsol_data->iparm[39] =  2;
  
  /* The number of row in global matrix, rhs element and solution vector
     that begins the input domain belonging to this MPI process */
  Bsol_data->iparm[40] = B->first_row;
  /* The number of row in global matrix, rhs element and solution vector
     that ends the input domain belonging to this MPI process   */
  Bsol_data->iparm[41] = B->first_row + B->nrow_local - 1;

  /* extract upper diag part of B */
  pevsl_Csr *U = &Bsol_data->U;
  pEVSL_ParcsrGetLocalMat(B, 0, NULL, U, 'U');

  MKL_INT n  = B->nrow_global;
  Bsol_data->n = n;
  MKL_INT *ia = U->ia;
  MKL_INT *ja = U->ja;
  double  *a  = U->a;

  /*
  for (i=0; i<=U->nrows; i++) {
    if (i>=510)
    printf("%d ", ia[i]);
  }
  printf("\n\n");
  */
  /* -------------------------------------------------------------------- */
  /* .. Reordering and Symbolic Factorization. This step also allocates   */
  /* all memory that is necessary for the factorization.                  */
  /* -------------------------------------------------------------------- */
  phase = 11;
  cluster_sparse_solver(Bsol_data->pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, 
                        &nrhs, Bsol_data->iparm, &msglvl, &ddum, &ddum, &commf, &error);
  if ( error != 0 ) {
    PEVSL_ABORT(MPI_COMM_WORLD, error, "\nERROR during symbolic factorization\n");
  }

  /* -------------------------------------------------------------------- */
  /* .. Numerical factorization.                                          */
  /* -------------------------------------------------------------------- */
  phase = 22;
  cluster_sparse_solver(Bsol_data->pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, 
                        &nrhs, Bsol_data->iparm, &msglvl, &ddum, &ddum, &commf, &error);
  if ( error != 0 ) {
    PEVSL_ABORT(MPI_COMM_WORLD, error, "\nERROR during numerical factorization\n");
  }
  
  *data = (void *) Bsol_data;

  return 0;
}

/** @brief Solver function of B with Pardiso
 *
 * */
void BSolDirect(double *b, double *x, void *data) {

  BSolDataDirect *Bsol_data = (BSolDataDirect *) data;

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT mtype = 2;       /* Real SPD matrix */
  MKL_INT msglvl = 0;
  MKL_INT error = 0;
  /* Integer dummy. */
  MKL_INT idum;
  /* Number of right hand sides. */
  MKL_INT nrhs = 1;
 
  MKL_INT n = Bsol_data->n;
  pevsl_Csr *U = &Bsol_data->U;
  MKL_INT *ia = U->ia;
  MKL_INT *ja = U->ja;
  double  *a  = U->a;
  MPI_Fint commf = Bsol_data->commf;

  /* --------------------------------------------- */
  /* .. Back substitution and iterative refinement.*/
  /* --------------------------------------------- */
  MKL_INT phase = 33;
  cluster_sparse_solver(Bsol_data->pt, &maxfct, &mnum, &mtype, &phase,
                        &n, a, ia, ja, &idum, &nrhs, Bsol_data->iparm, 
                        &msglvl, b, x, &commf, &error);
  if ( error != 0 ) {
    PEVSL_ABORT(MPI_COMM_WORLD, error, "\nERROR during solution\n");
  }
}

/** @brief Solver function of B with MUMPS
 *
 * */
void LTSolDirect(double *b, double *x, void *data) {
 
  BSolDataDirect *Bsol_data = (BSolDataDirect *) data;

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT mtype = 2;       /* Real SPD matrix */
  MKL_INT msglvl = 0;
  MKL_INT error = 0;
  /* Integer dummy. */
  MKL_INT idum;
  /* Number of right hand sides. */
  MKL_INT nrhs = 1;
 
  MKL_INT n = Bsol_data->n;
  pevsl_Csr *U = &Bsol_data->U;
  MKL_INT *ia = U->ia;
  MKL_INT *ja = U->ja;
  double  *a  = U->a;
  MPI_Fint commf = Bsol_data->commf;

  /* --------------------------------------------- */
  /* .. Back substitution and iterative refinement.*/
  /* --------------------------------------------- */
  MKL_INT phase = 333;
  cluster_sparse_solver(Bsol_data->pt, &maxfct, &mnum, &mtype, &phase,
                        &n, a, ia, ja, &idum, &nrhs, Bsol_data->iparm, 
                        &msglvl, b, x, &commf, &error);
  if ( error != 0 ) {
    PEVSL_ABORT(MPI_COMM_WORLD, error, "\nERROR during solution\n");
  }
}

/** @brief Free the factorization of B with Pardiso
 * */
void FreeBSolDirectData(void *data) {

  BSolDataDirect *Bsol_data = (BSolDataDirect *) data;

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT mtype = 2;       /* Real SPD matrix */
  MKL_INT msglvl = 0;
  MKL_INT error = 0;
  /* Double dummy */
  double ddum;
  /* Integer dummy. */
  MKL_INT idum;
  /* Number of right hand sides. */
  MKL_INT nrhs = 1;
 
  MKL_INT n = Bsol_data->n;
  pevsl_Csr *U = &Bsol_data->U;
  MKL_INT *ia = U->ia;
  MKL_INT *ja = U->ja;
  MPI_Fint commf = Bsol_data->commf;

  /* -------------------------------------------------------------------- */
  /* .. Termination and release of memory. */
  /* -------------------------------------------------------------------- */
  MKL_INT phase = -1; /* Release internal memory. */
  cluster_sparse_solver(Bsol_data->pt, &maxfct, &mnum, &mtype, &phase,
                        &n, &ddum, ia, ja, &idum, &nrhs, Bsol_data->iparm, 
                        &msglvl, &ddum, &ddum, &commf, &error);
  if ( error != 0 ) {
    PEVSL_ABORT(MPI_COMM_WORLD, error, "\nERROR during termination\n");
  }

  pEVSL_FreeCsr(U);
  PEVSL_FREE(data);
}

