#include "pevsl_protos.h"

/** \brief global variable of pEVSL
 *
 * global variable is guaranteed to be initialized
 * */
pevsl_Data pevsl_data;
pevsl_Stat pevsl_stat;

/**
 * @brief Initialize pEVSL  
 *
 * */
int pEVSL_Start() {

  /* make sure these are zeroed out */
  memset(&pevsl_stat, 0, sizeof(pevsl_Stat));
  memset(&pevsl_data, 0, sizeof(pevsl_Data));

  /* use MPI_COMM_WORLD rank as rand seed */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(rank);

  return 0;
}

/**
 * @brief Finish pEVSL
 *
 * */
int pEVSL_Finish() {
  if (pevsl_data.Amv) {
    PEVSL_FREE(pevsl_data.Amv);
  }
  if (pevsl_data.Bmv) {
    PEVSL_FREE(pevsl_data.Bmv);
  }
  if (pevsl_data.Bsol) {
    PEVSL_FREE(pevsl_data.Bsol);
  }
  if (pevsl_data.LTsol) {
    PEVSL_FREE(pevsl_data.LTsol);
  }

  return 0;
}

/** 
 * @brief Set the matrix A
 * 
 * */
int pEVSL_SetAParcsr(pevsl_Parcsr *A) {
  pevsl_data.N = A->ncol_global;
  pevsl_data.n = A->ncol_local;
  //pevsl_data.nfirst = A->first_col;
  if (!pevsl_data.Amv) {
    PEVSL_CALLOC(pevsl_data.Amv, 1, pevsl_Matvec);
  }
  pevsl_data.Amv->func = pEVSL_ParcsrMatvec;
  pevsl_data.Amv->data = (void *) A;
  
  return 0;
}

/**
 * @brief Set the user-input matvec routine and the associated data for A.
 * Save them in evsldata
 * @warning Once this matvec func is set, matrix A will be ignored even it
 * is provided
 * */
int pEVSL_SetAMatvec(int N, int n, MVFunc func, void *data) {
  pevsl_data.N = N;
  pevsl_data.n = n;
  //  pevsl_data.nfirst = ;
  if (!pevsl_data.Amv) {
    PEVSL_CALLOC(pevsl_data.Amv, 1, pevsl_Matvec);
  }
  pevsl_data.Amv->func = func;
  pevsl_data.Amv->data = data;
  
  return 0;
}

