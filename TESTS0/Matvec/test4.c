#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include "pevsl.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and test parcsrmv
------------------------------------------------------------*/
  int nrow, ncol, nnzrow, nnz, i, j, ngroups, flg, np, rank, *row_starts, *col_starts, seed;
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
  /*-------------------- matrix Alocal: csr format */
  pevsl_Csr Aloc;
  /*-------------------- default values */
  nrow = 64;
  ncol = 32;
  nnzrow = 17;
  ngroups = 2;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nrow [int] -ncol [int] -nnzrow [int] -ngroups [int] \n");
    return 0;
  }
  flg = findarg("nrow",    INT, &nrow,    argc, argv);
  flg = findarg("ncol",    INT, &ncol,    argc, argv);
  flg = findarg("nnzrow",  INT, &nnzrow,  argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  /*-------------------- Partition MPI_COMM_WORLD into ngroups subcomm's,
   * create a communicator for each group, and one for group leaders
   * saved in comm */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  /* each group should have the same starts */
  if (comm.group_rank == 0) {
    seed = time(NULL) + comm.group_id;
  }
  MPI_Bcast(&seed, 1, MPI_INT, 0, comm.comm_group);
  /* everyone in the group must have the same seed */
  srand(seed);

  /* row starts */
  row_starts = (int *) malloc((comm.group_size+1)*sizeof(int));
  row_starts[0] = 0;
  j = nrow;
  for (i=0; i<comm.group_size-1; i++) {
    /* size of rank i */
    int si = rand() % j;
    row_starts[i+1] = row_starts[i] + si;
    j -= si;
  }
  row_starts[comm.group_size] = nrow;
 
  /* col starts */
  col_starts = (int *) malloc((comm.group_size+1)*sizeof(int));
  col_starts[0] = 0;
  j = ncol;
  for (i=0; i<comm.group_size-1; i++) {
    /* size of rank i */
    int si = rand() % j;
    col_starts[i+1] = col_starts[i] + si;
    j -= si;
  }
  col_starts[comm.group_size] = ncol;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrix A
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 3rd and 4th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  pEVSL_ParcsrCreate(nrow, ncol, row_starts, col_starts, &A, comm.comm_group);
  // THE FOLLOWING COMBINATIONS SHOULD ALSO WORK
  //pEVSL_ParcsrCreate(nrow, ncol, NULL, NULL, &A, comm.comm_group);
  //pEVSL_ParcsrCreate(nrow, ncol, NULL, col_starts, &A, comm.comm_group);
  //pEVSL_ParcsrCreate(nrow, ncol, row_starts, NULL, &A, comm.comm_group);
  /* row/col range */
  /*
  PEVSL_SEQ_BEGIN(comm.comm_global);
  printf("row: [%d, %d), col: [%d, %d)\n", A.first_row, A.first_row + A.nrow_local,
                                           A.first_col, A.first_col + A.ncol_local);
  PEVSL_SEQ_END(comm.comm_global);
  */
  /*------------------- Create local Random CSR matrix */
  SpRandCsr(A.nrow_local, A.ncol_global, nnzrow, &Aloc);
  //printf("Aloc nnz %d\n", Aloc.ia[Aloc.nrows]);
  /*------------------- Setup Parcsr A */
  pEVSL_ParcsrSetup(&Aloc, &A);
  /*------------------- nnz(A) */
  nnz = pEVSL_ParcsrNnz(&A);
  /*------------------- Create parallel vector */
  /* x has the same parallel layout of A col */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &x);
  /* y has the same parallel layout of A row */
  pEVSL_ParvecCreate(A.nrow_global, A.nrow_local, A.first_row, comm.comm_group, &y);
  /* initialize x */
  for (i=0; i<x.n_local; i++) {
    x.data[i] = sin(x.n_first+i);
  }
  /* y = A * x */
  pEVSL_ParcsrMatvec(&A, &x, &y);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - -
   *   local csr matvec for reference
   * - - - - - - - - - - - - - - - - - - - - - - - - - */
  xseq = (double *) malloc(ncol*sizeof(double));
  yloc = (double *) malloc(y.n_local*sizeof(double));
  for (i=0; i<ncol; i++) {
    xseq[i] = sin(i);
  }
  pEVSL_Matvec(&Aloc, xseq, yloc);
  
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
    fprintf(stdout, "Random A : nrow %d x ncol %d, nnz %d\n", nrow, ncol, nnz);
  }
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("Group %d, size %d, ", comm.group_id, comm.group_size);
    printf("row_starts:");
    for (i=0; i<comm.group_size+1; i++) {
      printf(" %d", row_starts[i]);
    }
    printf("; ");
    printf("col_starts:");
    for (i=0; i<comm.group_size+1; i++) {
      printf(" %d", col_starts[i]);
    }
    printf("\n");
    printf("Matvec ||SEQ-PARA|| = %e\n", err_all);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&x);
  pEVSL_ParvecFree(&y);
  pEVSL_FreeCsr(&Aloc);
  free(xseq);
  free(yloc);
  free(row_starts);
  free(col_starts);
  CommInfoFree(&comm);

  MPI_Finalize();

  return 0;
}

