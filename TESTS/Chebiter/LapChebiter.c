#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"
#include "pevsl_mumps.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts_in, MPI_Comm comm);
void PEVSL_FORT(pevsl_chebiter)(int *type, double *b, double *x, uintptr_t *chebf90);

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
  int n, nx, ny, nz, i, deg, /*j, npts, nvec, Mdeg, nev, */
      ngroups, mlan, msteps, flg, np, rank;
  double lmax, lmin, tol;
  //double *xdos, *ydos;
  /*-------------------- communicator struct, which contains all the communicators */
  CommInfo comm;
  pevsl_Parvec vinit;
  /*--------------------- Initialize MPI */
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  /*-------------------- size and rank */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*-------------------- matrices A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- default values */
  nx = 8;
  ny = 8;
  nz = 8;
  ngroups = 2;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -nslices [int] -a [double] -b [double]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  /*-------------------- matrix size */
  n = nx * ny * nz;
  /*-------------------- Create communicators for groups, group leaders */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  /*-------------------- output the problem settings */
  if (!comm.group_rank) {
    fprintf(stdout, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
  }
  /*-------------------- generate 1D/3D Laplacian Parcsr matrices */
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrices A and B
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx*ny*nz, 1, 1, NULL, comm.comm_group);
  /*-------------------- start pEVSL */
  pEVSL_Start(argc, argv);
  /*-------------------- set the left-hand side matrix A */
  pEVSL_SetAParcsr(&A);
  /* compute bounds */
  mlan   = 50;
  msteps = 200;
  tol    = 1e-8;
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
  pEVSL_ParvecRand(&vinit);
  pEVSL_LanTrbounds(mlan, msteps, tol, &vinit, 1, &lmin, &lmax, comm.comm_group, NULL);
  /* pEVSL done */
  pEVSL_Finish();

  pevsl_Parvec bb, xx, dd;
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &bb);
  pEVSL_ParvecRand(&bb);
  pEVSL_ParvecDupl(&bb, &xx);
  pEVSL_ParvecDupl(&bb, &dd);

  //pEVSL_SetBSol(pEVSL_ChebIterSolv2, &BsolCheb);
  //pevsl_bsv_f90_(bb.data, xx.data);
  //
  Chebiter_Data *Cheb;
  int type = 2;
  deg = 10;
#if 0
  uintptr_t chebf90, Af90;
  MPI_Fint Fcomm = MPI_Comm_c2f(comm.comm_group);
  Af90 = (uintptr_t) &A;
  pevsl_setup_chebiter_f90_(&lmin, &lmax, &deg, &Af90, &Fcomm, &chebf90);
  pevsl_chebiter_f90_(&type, bb.data, xx.data, &chebf90);
  Cheb = (Chebiter_Data *) chebf90;
#else
  PEVSL_MALLOC(Cheb, 1, Chebiter_Data);
  pEVSL_ChebIterSetup(lmin, lmax, deg, &A, comm.comm_group, Cheb);
  pEVSL_ChebIterSolv2(bb.data, xx.data, (void *) Cheb);
#endif
  printf("eig A: [%e,%e]\n", Cheb->lb, Cheb->ub);
  if (Cheb->res) {
    printf("CHEB ITER RES\n");
    for (i=0; i<Cheb->deg+1; i++) {
      printf("i %3d: %e\n", i, Cheb->res[i]);
    }
  }

  double resnorm;
  pEVSL_ParcsrMatvec(&A, &xx, &dd);
  pEVSL_ParvecAxpy(-1.0, &bb, &dd);
  pEVSL_ParvecNrm2(&dd, &resnorm);
  printf("res %e\n", resnorm);
  
  pEVSL_ChebIterFree(Cheb);
  free(Cheb);

  MPI_Barrier(MPI_COMM_WORLD);

  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&vinit);
  pEVSL_ParvecFree(&bb);
  pEVSL_ParvecFree(&xx);
  pEVSL_ParvecFree(&dd);

  CommInfoFree(&comm);
  MPI_Finalize();

  return 0;
}

