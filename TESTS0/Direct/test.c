#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"
#include "pevsl_direct.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh
  and test direct sovlers 
------------------------------------------------------------*/
  int n, nx, ny, nz, ngroups, flg, np, rank;
  double err, nrmb;
  double tme, tms, tfact, tsolv;
  /*-------------------- communicator struct, which contains all the communicators */
  CommInfo comm;
  pevsl_Parvec b, x, r;
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
  /*-------------------- Asol */
  void *Asol;
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
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
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
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  /*-------------------- use MUMPS as the solver for B */
  tms = MPI_Wtime(); 
  SetupBSolDirect(&A, &Asol);
  tme = MPI_Wtime();
  tfact = tme - tms;

  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &b);
  pEVSL_ParvecDupl(&b, &x);
  pEVSL_ParvecDupl(&b, &r);
  pEVSL_ParvecRand(&b);
 
  tms = MPI_Wtime(); 
  BSolDirect(b.data, x.data, Asol);
  tme = MPI_Wtime();
  tsolv = tme - tms;

  pEVSL_ParcsrMatvec(&A, &x, &r);
  pEVSL_ParvecAxpy(-1.0, &b, &r);
  pEVSL_ParvecNrm2(&r, &err);
  pEVSL_ParvecNrm2(&b, &nrmb);

  if (comm.global_rank == 0) {
    fprintf(stdout, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
  }
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    fprintf(stdout, "Group %d, size %d\n", comm.group_id, comm.group_size);
    printf("T-fact %f\n", tfact);
    printf("T-solv %f\n", tsolv);
    printf("res norm = %e\n", err / nrmb);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  FreeBSolDirectData(Asol);
  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&b);
  pEVSL_ParvecFree(&x);
  pEVSL_ParvecFree(&r);
  CommInfoFree(&comm);
  MPI_Finalize();

  return 0;
}

