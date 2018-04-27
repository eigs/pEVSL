#include "pevsl_int.h"
#include "pevsl_direct.h"
#include "dmumps_c.h"

/**
 * @file pevsl_mumps.c
 * @brief Definitions used for MUMPS interface
 */

typedef struct _BSolDataDirect {
  /* global and local size */
  int N, n;
  DMUMPS_STRUC_C solver;
  double *rhs_global;
  /* number of local entries and start of local entries
   * only exists on rank 0, needed by MUMPS since we need to 
   * centralize rhs that needs to do Gatherv */
  int *ncols;
  int *icols;
} BSolDataDirect;

/* MUMPS macro s.t. indices match documentation */
#define ICNTL(I) icntl[(I)-1]

/** @brief Setup the B-sol by factorization with MUMPS
 *
 * @param[in] B parcsr matrix B
 * @param[out] data      Output B-sol struct
 * */
int SetupBSolDirect(pevsl_Parcsr *B, void **data) {
  int i, nglobal, nlocal, rank, size;
  BSolDataDirect *Bsol_data;
  PEVSL_MALLOC(Bsol_data, 1, BSolDataDirect);

  /* MUMPS solver will use the same communicator as B */
  MPI_Comm comm = B->comm;
  /*-------------------- MPI rank and size in comm */
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  /* global and local sizes */
  nglobal = B->nrow_global;
  nlocal = B->nrow_local;
  Bsol_data->N = nglobal;
  Bsol_data->n = nlocal;
  /* check if sizes match */
  //PEVSL_CHKERR(nglobal != pevsl_data.N);
  //PEVSL_CHKERR(nlocal != pevsl_data.n);
  /* create solver the communicator of B */
  Bsol_data->solver.comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm);
  Bsol_data->solver.par = 1; /* host is also involved */
  Bsol_data->solver.sym = 1; /* 0: nonsymmetric, 1:SPD, 2: symmetric */
  Bsol_data->solver.job = -1; /* initialization */
  dmumps_c(&(Bsol_data->solver));
  /*------------------- local coo matrix (1-based indices) 
   *                    Upper triangular part */
  pevsl_Coo coo_local;
  pEVSL_ParcsrGetLocalMat(B, 1, &coo_local, NULL, 'U');
  /* adjust the row indices of COO to global indices */
  for (i=0; i<coo_local.nnz; i++) {
    /* NOTE: change row indices to global ones */
    coo_local.ir[i] += B->first_row;
  }
  Bsol_data->solver.ICNTL(2) = -1; /* output suppressed */
  Bsol_data->solver.ICNTL(3) = -1; /* output suppressed */
  Bsol_data->solver.ICNTL(18) = 3; /* distributed matrix */
  Bsol_data->solver.ICNTL(28) = 2; /* parallel ordering */
  Bsol_data->solver.ICNTL(29) = 2; /* parmetis */
  Bsol_data->solver.n = nglobal;
  Bsol_data->solver.nnz_loc = coo_local.nnz;
  Bsol_data->solver.irn_loc = coo_local.ir;
  Bsol_data->solver.jcn_loc = coo_local.jc;
  Bsol_data->solver.a_loc = coo_local.vv;
  Bsol_data->solver.job = 4; /* analysis and factorization */
  dmumps_c(&(Bsol_data->solver));
  /* free local coo */
  pEVSL_FreeCoo(&coo_local);
  /* setup for the solve */
  Bsol_data->rhs_global = NULL;
  Bsol_data->ncols = NULL;
  Bsol_data->icols = NULL;
  /* global rhs on the root */
  if (rank == 0) {
    PEVSL_MALLOC(Bsol_data->rhs_global, nglobal, double);
    PEVSL_MALLOC(Bsol_data->ncols, size, int);
    PEVSL_MALLOC(Bsol_data->icols, size, int);
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
      Bsol_data->ncols[i] = j2-j1;
      Bsol_data->icols[i] = j1;
    }
  }

  *data = (void *) Bsol_data;

  return 0;
}


/** @brief Solver function of B with MUMPS
 *
 * */
void BSolDirect(double *b, double *x, void *data) {
  /* MUMPS data */
  BSolDataDirect *mumps_data = (BSolDataDirect *)data;
  /* MPI communicator */
  MPI_Comm comm = MPI_Comm_f2c(mumps_data->solver.comm_fortran);
  /* MPI rank */
  int rank;
  MPI_Comm_rank(comm, &rank);
  int *ncols = mumps_data->ncols;
  int *icols = mumps_data->icols;
  /* local size */
  //int nlocal = pevsl_data.n;
  int nlocal = mumps_data->n;
  /* MUMPS needs rhs to be centralized */
  double *rhs_global = mumps_data->rhs_global;
  /* gather rhs to rank 0 */
  MPI_Gatherv(b, nlocal, MPI_DOUBLE, rhs_global,
              ncols, icols, MPI_DOUBLE, 0, comm);
  /*----------------- solve */
  mumps_data->solver.rhs = rhs_global;
  mumps_data->solver.job = 3;
  dmumps_c(&(mumps_data->solver));
  /*----------------- check for error */
  if (mumps_data->solver.infog[0] != 0) {
    char errmsg[1024];
    sprintf(errmsg, " (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
            0, mumps_data->solver.infog[0], mumps_data->solver.infog[1]);
    PEVSL_ABORT(comm, 1, errmsg);
  }
  /*----------------- distribute the solution */
  MPI_Scatterv(rhs_global, ncols, icols, MPI_DOUBLE, x, nlocal,
               MPI_DOUBLE, 0, comm);
}

/** @brief Solver function of B with MUMPS
 *
 * */
void LTSolDirect(double *b, double *x, void *data) {
  PEVSL_ABORT(MPI_COMM_WORLD, PEVSL_NOT_IMPLEMENT, "MUMPS LT-SOL NOT IMPLEMENTED!");
}

/** @brief Free the factorization of B with MUMPS
 *
 * @param data  MUMPS solver instance
 * */
void FreeBSolDirectData(void *data) {
  BSolDataDirect *Bsol_data = (BSolDataDirect *) data;
  /* destroy the solver */
  Bsol_data->solver.job = -2;
  dmumps_c(&(Bsol_data->solver));

  if (Bsol_data->rhs_global) {
    PEVSL_FREE(Bsol_data->rhs_global);
  }
  if (Bsol_data->ncols) {
    PEVSL_FREE(Bsol_data->ncols);
  }
  if (Bsol_data->icols) {
    PEVSL_FREE(Bsol_data->icols);
  }

  PEVSL_FREE(data);
}


