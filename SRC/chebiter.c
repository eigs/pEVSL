#include "pevsl_int.h"

/** @brief Solver function to setup Chebyshev iterations
 *
 * */
int pEVSL_SetupChebIterMatB(int deg, int lanm, int msteps, double tol, 
                            MPI_Comm comm, BSolDataChebiter *data) {
  int N = pevsl_data.N;
  int n = pevsl_data.n;
  int nfirst = pevsl_data.nfirst;
  /* save the states, will be changed in this function */
  int ifGenEv = pevsl_data.ifGenEv;
  pevsl_Matvec *Amvsave = pevsl_data.Amv;
  pevsl_Parvec vinit;
  double lmin, lmax;
  
  pEVSL_ParvecCreate(N, n, nfirst, comm, &vinit);
  pEVSL_ParvecRand(&vinit);
  /* compute eig bounds of B */
  pEVSL_SetStdEig();
  pevsl_data.Amv = pevsl_data.Bmv;
  /* compute bounds */
  pEVSL_LanTrbounds(lanm, msteps, tol, &vinit, 1, &lmin, &lmax, comm, NULL);
  /* save the results */
  data->lb = lmin;
  data->ub = lmax;
  data->deg = deg;
  /* restore states */
  pevsl_data.ifGenEv = ifGenEv;
  pevsl_data.Amv = Amvsave;
  
  return 0;
}

/** @brief Solver function of B with Chebyshev iterations
 *
 * */
int pEVSL_ChebIterMatB(double *b, double *x, void *data) {
  BSolDataChebiter *Chebdata 
  return 0;
}
