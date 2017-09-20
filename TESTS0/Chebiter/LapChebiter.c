#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "pevsl_itsol.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char *argv[]) {
/*----------------------------------------------------------*
  generates a Laplacian matrix on an nx x ny x nz mesh, and
  test Chebyshev iterations for solving linear systems
 *----------------------------------------------------------*/
  int n, nx, ny, nz, i, deg, ngroups, mlan, msteps, flg, np, rank, type;
  double lmax, lmin, tol;
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
  /*-------------------- matrix A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- instance of pEVSL */
  pevsl_Data *pevsl;
  /*-------------------- default values */
  nx = 8;
  ny = 8;
  nz = 8;
  ngroups = 2;
  deg = 10;
  /* two types of Chebyshev iterations implemented */
  type = 2;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -ngroups [int] -type [int] -deg [int]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  flg = findarg("type", INT, &type, argc, argv);
  flg = findarg("deg", INT, &deg, argc, argv);
  /*-------------------- matrix size */
  n = nx * ny * nz;
  /*-------------------- Create communicators for groups, group leaders */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
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
  pEVSL_Start(comm.comm_group, &pevsl);
  /*-------------------- set the left-hand side matrix A */
  pEVSL_SetAParcsr(pevsl, &A);
  /* compute bounds */
  mlan   = 50;
  msteps = 200;
  tol    = 1e-8;
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
  pEVSL_ParvecRand(&vinit);
  pEVSL_LanTrbounds(pevsl, mlan, msteps, tol, &vinit, 1, &lmin, &lmax, NULL);
  /*-------------------- pEVSL done */
  pEVSL_Finish(pevsl);

  pevsl_Parvec bb, xx, dd;
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &bb);
  pEVSL_ParvecRand(&bb);
  pEVSL_ParvecDupl(&bb, &xx);
  pEVSL_ParvecDupl(&bb, &dd);

  void *Cheb;
  pEVSL_ChebIterSetup(lmin, lmax, deg, &A, &Cheb);
  if (type == 1) {
    pEVSL_ChebIterSolv1(bb.data, xx.data, Cheb);
  } else {
    pEVSL_ChebIterSolv2(bb.data, xx.data, Cheb);
  }
  double resnorm;
  pEVSL_ParcsrMatvec(&A, &xx, &dd);
  pEVSL_ParvecAxpy(-1.0, &bb, &dd);
  pEVSL_ParvecNrm2(&dd, &resnorm);

  if (comm.global_rank == 0) {
    fprintf(stdout, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
  }
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    fprintf(stdout, "Group %d, size %d\n", comm.group_id, comm.group_size);
    printf("eig(A): [%e,%e], degree: %d\n", lmin, lmax, deg);
    double *chebres = pEVSL_ChebIterGetRes(Cheb);
    if (chebres) {
      printf("CHEB ITER RES\n");
      for (i=0; i<deg+1; i++) {
        printf("%3d: %e\n", i, chebres[i]);
      }
    }
    printf("Computed final res = %.15e\n", resnorm);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  pEVSL_ChebIterStatsPrint(Cheb, stdout);
  pEVSL_ChebIterFree(Cheb);

  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&vinit);
  pEVSL_ParvecFree(&bb);
  pEVSL_ParvecFree(&xx);
  pEVSL_ParvecFree(&dd);

  CommInfoFree(&comm);
  MPI_Finalize();

  return 0;
}

