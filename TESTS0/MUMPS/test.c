#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "io.h"
#include "dmumps_c.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
  
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and test MUMPS
------------------------------------------------------------*/
  int n, nx, ny, nz, i, ngroups, flg, np, rank;
  double tme, tms, tfact, tsolv, tgatherb, tscatterx, tmv;
  /*-------------------- pEVSL communicator, which contains all the communicators */
  CommInfo comm;
  pevsl_Coo coo_local;
  pevsl_Parvec rhs, sol, res;
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
  nx = 8;
  ny = 8;
  nz = 8;
  ngroups = 2;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -ngroups [int]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  n = nx * ny * nz;
  /*-------------------- Create communicators for groups, group leaders */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
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
  pEVSL_ParcsrGetLocalMat(&A, 1, &coo_local, NULL, 'U');
  /* adjust the row indices of COO to global indices */
  for (i=0; i<coo_local.nnz; i++) {
    /* NOTE: change row indices to global ones */
    coo_local.ir[i] += A.first_row;
    //printf("%d %d %f\n", coo_local.ir[i], coo_local.jc[i], coo_local.vv[i]);
  }
  solver.ICNTL(2) = -1; /* output suppressed */ // 6: on screen
  solver.ICNTL(3) = -1; /* output suppressed */ // 6: on screen
  //solver.ICNTL(4) = 4;  /* output level */
  solver.ICNTL(18) = 3; /* distributed matrix */
  solver.ICNTL(28) = 2; /* parallel ordering */
  solver.ICNTL(29) = 2; /* parmetis */
  solver.n = n;
  solver.nnz_loc = coo_local.nnz;
  solver.irn_loc = coo_local.ir;
  solver.jcn_loc = coo_local.jc;
  solver.a_loc = coo_local.vv;
  solver.job = 4;
  tms = MPI_Wtime(); 
  dmumps_c(&solver);
  tme = MPI_Wtime(); 
  tfact = tme - tms;
  /*------------------- Create parallel vector: random rhs guess */
  //solver.ICNTL(20) = 0; /* dense rhs */
  //solver.ICNTL(21) = 0; /* centralized solution */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, A.comm, &rhs);
  //pEVSL_ParvecDupl(&rhs, &x);
  //pEVSL_ParvecSetScalar(&x, 1.0);
  //pEVSL_ParcsrMatvec(&A, &x, &rhs);
  srand(0);
  pEVSL_ParvecRand(&rhs);
  /*------------------- gather rhs to root */
  double *rhs_global = NULL;
  int *ncols = NULL, *icols = NULL;
  if (comm.group_rank == 0) {
    rhs_global = (double *) malloc(n*sizeof(double));
    ncols = (int *) malloc(comm.group_size*sizeof(int));
    icols = (int *) malloc(comm.group_size*sizeof(int));
    for (i=0; i<comm.group_size; i++) {
      int j1, j2;
      pEVSL_Part1d(n, comm.group_size, &i, &j1, &j2, 1);
      ncols[i] = j2-j1;
      icols[i] = j1;
    }
  }
  tms = MPI_Wtime(); 
  MPI_Gatherv(rhs.data, rhs.n_local, MPI_DOUBLE, rhs_global,
              ncols, icols, MPI_DOUBLE, 0, comm.comm_group);
  tme = MPI_Wtime();
  tgatherb = tme - tms;
  /*
  if (comm.group_rank == 0) {
    for (i=0; i<n; i++) {
      printf("%e\n", rhs_global[i]);
    }
  }
  */
  /*----------------- solve */
  solver.rhs = rhs_global;
  solver.job = 3;
  tms = MPI_Wtime(); 
  dmumps_c(&solver);
  tme = MPI_Wtime();
  tsolv = tme - tms;

  if (solver.infog[0] != 0) {
    printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
           0, solver.infog[0], solver.infog[1]);
  }

  /*
  if (comm.group_rank == 0) {
    for (i=0; i<n; i++) {
      printf("%e\n", rhs_global[i]);
    }
  }
  */
  /*----------------- distribute the solution */
  tms = MPI_Wtime(); 
  pEVSL_ParvecDupl(&rhs, &sol);
  MPI_Scatterv(rhs_global, ncols, icols, MPI_DOUBLE, sol.data, sol.n_local,
               MPI_DOUBLE, 0, comm.comm_group);

  tme = MPI_Wtime();
  tscatterx = tme - tms;

  if (comm.group_rank == 0) {
    free(rhs_global);
    free(ncols);
    free(icols);
  }

  //double nrmx;
  //pEVSL_ParvecNrm2(&sol, &nrmx);
  //printf("sol nrm %.15e\n", nrmx);

  /*----------------- check residual */
  double nrm, nrmb;
  pEVSL_ParvecNrm2(&rhs, &nrmb);
  //printf("rhs nrm %.15e\n", nrmb);
  pEVSL_ParvecDupl(&rhs, &res);
  tms = MPI_Wtime(); 
  pEVSL_ParcsrMatvec(&A, &sol, &res);
  tme = MPI_Wtime();
  tmv = tme - tms;
  pEVSL_ParvecAxpy(-1.0, &rhs, &res);
  pEVSL_ParvecNrm2(&res, &nrm);
  
  /*----------------- done */
  solver.job = -2;
  dmumps_c(&solver);

  if (comm.global_rank == 0) {
    fprintf(stdout, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
  }
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    fprintf(stdout, "Group %d, size %d\n", comm.group_id, comm.group_size);
    printf("T-fact       %f\n", tfact);
    printf("T-solv       %f\n", tsolv);
    printf("T-gatherb    %f\n", tgatherb);
    printf("T-scatterx   %f\n", tscatterx);
    printf("T-matvec     %f\n", tmv);
    printf("res norm     %e\n", nrm / nrmb);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }
  
  pEVSL_FreeCoo(&coo_local);
  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&rhs);
  pEVSL_ParvecFree(&sol);
  pEVSL_ParvecFree(&res);
  CommInfoFree(&comm);
  MPI_Finalize();

  return 0;
}

