#include "pevsl_int.h"
#include "pevsl_mumps.h"

/* MUMPS macro s.t. indices match documentation */
#define ICNTL(I) icntl[(I)-1]

/** @brief Setup the B-sol by factorization with MUMPS
 *
 * @param B      parcsr matrix B
 * */
int SetupBSolMumps(pevsl_Parcsr *B, BSolDataMumps *data) {
  int i, nglobal, nlocal, rank, size;
  /* MUMPS solver will use the same communicator as B */
  MPI_Comm comm = B->comm;
  /*-------------------- MPI rank and size in comm */
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  /* global and local sizes */
  nglobal = B->nrow_global;
  nlocal = B->nrow_local;
  data->N = nglobal;
  data->n = nlocal;
  /* check if sizes match */
  //PEVSL_CHKERR(nglobal != pevsl_data.N);
  //PEVSL_CHKERR(nlocal != pevsl_data.n);
  /* create solver the communicator of B */
  data->solver.comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm);
  data->solver.par = 1; /* host is also involved */
  data->solver.sym = 1; /* 0: nonsymmetric, 1:SPD, 2: symmetric */
  data->solver.job = -1; /* initialization */
  dmumps_c(&(data->solver));
  /*------------------- local coo matrix (1-based indices) 
   *                    Upper triangular part */
  pevsl_Coo coo_local;
  pEVSL_ParcsrGetLocalMat(B, 1, &coo_local, NULL, 'U');
  /* adjust the row indices of COO to global indices */
  for (i=0; i<coo_local.nnz; i++) {
    /* NOTE: change row indices to global ones */
    coo_local.ir[i] += B->first_row;
  }
  data->solver.ICNTL(2) = -1; /* output suppressed */
  data->solver.ICNTL(3) = -1; /* output suppressed */
  data->solver.ICNTL(18) = 3; /* distributed matrix */
  data->solver.ICNTL(28) = 2; /* parallel ordering */
  data->solver.ICNTL(29) = 2; /* parmetis */
  data->solver.n = nglobal;
  data->solver.nnz_loc = coo_local.nnz;
  data->solver.irn_loc = coo_local.ir;
  data->solver.jcn_loc = coo_local.jc;
  data->solver.a_loc = coo_local.vv;
  data->solver.job = 4; /* analysis and factorization */
  dmumps_c(&(data->solver));
  /* free local coo */
  pEVSL_FreeCoo(&coo_local);
  /* setup for the solve */
  data->rhs_global = NULL;
  data->ncols = NULL;
  data->icols = NULL;
  /* global rhs on the root */
  if (rank == 0) {
    PEVSL_MALLOC(data->rhs_global, nglobal, double);
    PEVSL_MALLOC(data->ncols, size, int);
    PEVSL_MALLOC(data->icols, size, int);
    for (i=0; i<size; i++) {
      int j1, j2;
      /* range of each proc */
      if (B->col_starts) {
        /* avaiable from Parcsr */
        j1 = B->col_starts[i];
        j2 = B->col_starts[i+1];
      } else {
        /* if NULL, use default partition */
        pEVSL_Part1d(nglobal, size, &i, &j1, &j2, 1);
      }
      data->ncols[i] = j2-j1;
      data->icols[i] = j1;
    }
  }

  return 0;
}

/** @brief Free the factorization of B with MUMPS
 *
 * @param data  MUMPS solver instance
 * */
int FreeBSolMumps(BSolDataMumps *data) {
  /* free the solver */
  data->solver.job = -2;
  dmumps_c(&(data->solver));

  if (data->rhs_global) {
    free(data->rhs_global);
  }
  if (data->ncols) {
    free(data->ncols);
  }
  if (data->icols) {
    free(data->icols);
  }

  return 0;
}

/** @brief Solver function of B with MUMPS
 *
 * */
void BSolMumps(double *b, double *x, void *data, MPI_Comm comm) {
  /* MUMPS data */
  BSolDataMumps *mumps_data = (BSolDataMumps *)data;
  /* MPI communicator */
  MPI_Comm comm2 = MPI_Comm_f2c(mumps_data->solver.comm_fortran);
  /* MPI rank */
  int rank;
  MPI_Comm_rank(comm2, &rank);
  int *ncols = mumps_data->ncols;
  int *icols = mumps_data->icols;
  /* local size */
  //int nlocal = pevsl_data.n;
  int nlocal = mumps_data->n;
  /* MUMPS needs rhs to be centralized */
  double *rhs_global = mumps_data->rhs_global;
  /* gather rhs to rank 0 */
  MPI_Gatherv(b, nlocal, MPI_DOUBLE, rhs_global,
              ncols, icols, MPI_DOUBLE, 0, comm2);
  /*----------------- solve */
  mumps_data->solver.rhs = rhs_global;
  mumps_data->solver.job = 3;
  dmumps_c(&(mumps_data->solver));
  /*----------------- check for error */
  if (mumps_data->solver.infog[0] != 0) {
    char errmsg[1024];
    sprintf(errmsg, " (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
            0, mumps_data->solver.infog[0], mumps_data->solver.infog[1]);
    PEVSL_ABORT(comm2, 1, errmsg);
  }
  /*----------------- distribute the solution */
  MPI_Scatterv(rhs_global, ncols, icols, MPI_DOUBLE, x, nlocal,
               MPI_DOUBLE, 0, comm2);
}

