#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and test parcsrmv
------------------------------------------------------------*/
  int n, nx, ny, nz, i, ngroups, flg, np, rank;
  double err, err_all;
  /*-------------------- communicator struct, which contains all the communicators */
  CommInfo comm;
  /* vector x and y */
  pevsl_Parvec x, y;
  double *xseq, *yloc, t;
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
  /*-------------------- matrix A: seq csr format */
  pevsl_Csr Aseq;
  /*-------------------- default values */
  nx = 16;
  ny = 16;
  nz = 16;
  ngroups = 2;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -ngroups [int] \n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  /* global matrix size */
  n = nx * ny * nz;
  /*-------------------- Partition MPI_COMM_WORLD into ngroups subcomm's,
   * create a communicator for each group, and one for group leaders
   * saved in comm */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrix A
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  /*------------------- Create parallel vector */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &x);
  pEVSL_ParvecDupl(&x, &y);
  /* initialize x */
  for (i=0; i<x.n_local; i++) {
    x.data[i] = sin(x.n_first+i);
  }
  /* y = A * x */
  pEVSL_ParcsrMatvec(&A, &x, &y);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - -
   *   seqential matvec for reference
   * - - - - - - - - - - - - - - - - - - - - - - - - - */
  LocalLapGen(nx, ny, nz, y.n_first, y.n_first+y.n_local, &Aseq);
  xseq = (double *) malloc(n*sizeof(double));
  yloc = (double *) malloc(y.n_local*sizeof(double));
  for (i=0; i<n; i++) {
    xseq[i] = sin(i);
  }
  pEVSL_Matvec(&Aseq, xseq, yloc);
  
  /* check result: each rank check its own portion 
   * diff between y and yloc */
  err = 0.0;
  for (i=0; i<y.n_local; i++) {
    t = y.data[i] - yloc[i];
    err += t*t;
  }
  err = sqrt(err);
  MPI_Reduce(&err, &err_all, 1, MPI_DOUBLE, MPI_SUM, 0, comm.comm_group);
  
  if (comm.global_rank == 0) {
    fprintf(stdout, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
  }
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("Group %d, size %d: Matvec ||SEQ-PARA|| = %e\n", 
           comm.group_id, comm.group_size, err_all);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&x);
  pEVSL_ParvecFree(&y);
  pEVSL_FreeCsr(&Aseq);
  free(xseq);
  free(yloc);
  CommInfoFree(&comm);

  MPI_Finalize();

  return 0;
}

