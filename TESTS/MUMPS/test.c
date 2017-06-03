#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "io.h"
#include "dmumps_c.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
  
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */

int lapgen(int nx, int ny, int nz, int m1, int m2, pevsl_Coo *coo, pevsl_Csr *csr);
int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts_in, MPI_Comm comm);

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and computes all eigenvalues in a given interval [a  b]
  The default set values are
  nx = 41; ny = 53; nz = 1;
  a = 0.4; b = 0.8;
  nslices = 1 [one slice only] 
  other parameters 
  tol [tolerance for stopping - based on residual]
  Mdeg = pol. degree used for DOS
  nvec  = number of sample vectors used for DOS
  This uses:
  Non-restart Lanczos with polynomial filtering
------------------------------------------------------------*/
  int n, nx, ny, nz, i, j, npts, nslices, nvec, Mdeg, nev, 
      ngroups, mlan, ev_int, sl, flg, ierr, np, rank;
  /* find the eigenvalues of A in the interval [a,b] */
  /*-------------------- pEVSL communicator, which contains all the communicators */
  pevsl_Comm comm;
  pevsl_Coo coo_local;
  pevsl_Parvec rhs;
  pevsl_Polparams pol;
  FILE *fstats = NULL;
  /*--------------------- Initialize MPI */
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  /*-------------------- size and rank */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*-------------------- matrix A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- default values */
  nx = 16;
  ny = 16;
  nz = 16;
  ngroups = 3;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  /*-------------------- eigenvalue bounds set by hand */
  n = nx * ny * nz;
  /*-------------------- */
  /*-------------------- start pEVSL */
  pEVSL_Start(argc, argv);
  /*-------------------- Create communicators for groups, group leaders */
  pEVSL_CommCreate(&comm, MPI_COMM_WORLD, ngroups);
  /*-------------------- Group leader (group_rank == 0) creates output file */
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrix A
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  /*-------------------- MUMPS */
  DMUMPS_STRUC_C solver;
  solver.comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm.comm_group);
  solver.par = 1; /* host is also involved */
  solver.sym = 1; /* SPD */
  solver.job = -1; /* initialization */
  dmumps_c(&solver);
  /*------------------- local coo matrix (1-based indices) */
  pEVSL_ParcsrGetLocalMat(&A, 1, &coo_local, NULL);
  solver.ICNTL(18) = 3; /* distributed matrix */
  //solver.ICNTL(28) = 2; /* parallel ordering */
  //solver.ICNTL(29) = 2; /* parmetis */
  solver.n = n;
  //solver.nnz = pEVSL_ParcsrNnz(&A);
  solver.nnz_loc = pEVSL_ParcsrLocalNnz(&A);
  solver.irn_loc = coo_local.ir;
  solver.jcn_loc = coo_local.jc;
  solver.A_loc = coo_local.vv;
  solver.job = 4;
  dmumps_c(&solver);
  /*------------------- Create parallel vector: random rhs guess */
  //solver.ICNTL(20) = 0; /* dense rhs */
  //solver.ICNTL(21) = 0; /* centralized solution */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, A.comm, &rhs);
  pEVSL_ParvecRand(&rhs);
  /*------------------- gather rhs to root */
  double *rhs_global = (double *) malloc(n*sizeof(double));
  if (comm.group_rank == 0) {
  }
  

  if (fstats) fclose(fstats);
  pEVSL_FreeCoo(&coo_local);
  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&vinit);

  pEVSL_CommFree(&comm);

  pEVSL_Finish();

  MPI_Finalize();

  return 0;
}

