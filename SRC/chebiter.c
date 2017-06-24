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
  pevsl_Parvec *vinit, *r, *p;
  double lmin, lmax;
  
  PEVSL_MALLOC(vinit, 1, pevsl_Parvec);
  pEVSL_ParvecCreate(N, n, nfirst, comm, vinit);
  pEVSL_ParvecRand(vinit);
  /* compute eig bounds of B */
  pEVSL_SetStdEig();
  pevsl_data.Amv = pevsl_data.Bmv;
  /* compute bounds */
  pEVSL_LanTrbounds(lanm, msteps, tol, vinit, 1, &lmin, &lmax, comm, NULL);
  /* save the results */
  data->lb = lmin;
  data->ub = lmax;
  data->deg = deg;
  /* restore states */
  pevsl_data.ifGenEv = ifGenEv;
  pevsl_data.Amv = Amvsave;
  
  /* alloc work space */
  PEVSL_MALLOC(r, 1, pevsl_Parvec);
  PEVSL_MALLOC(p, 1, pevsl_Parvec);
  pEVSL_ParvecDupl(vinit, r);
  pEVSL_ParvecDupl(vinit, p);
  data->w = vinit; /* reuse vinit */
  data->r = r;
  data->p = p;

  return 0;
}

/** @brief Solver function of B with Chebyshev iterations
 *
 * */
void pEVSL_ChebIterSolMatB(double *db, double *dx, void *data, MPI_Comm comm) {
  int i;
  /* sizes */
  int N = pevsl_data.N;
  int n = pevsl_data.n;
  int nfirst = pevsl_data.nfirst;
  /* Cheb sol data */
  BSolDataChebiter *Chebdata = (BSolDataChebiter *) data;
  double d, c, alp=0.0, bet, norm_r0, norm_r, t;
  pevsl_Parvec *w = Chebdata->w;
  pevsl_Parvec *r = Chebdata->r;
  pevsl_Parvec *p = Chebdata->p;
  /* Parvec wrapper */
  pevsl_Parvec b, x;
  int deg = Chebdata->deg;

  /* center and half width */
  d = (Chebdata->ub + Chebdata->lb) * 0.5;
  c = (Chebdata->ub - Chebdata->lb) * 0.5;
  /* wrap b and x into pevsl_Parvec */
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &b, db);
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &x, dx);
  /* use 0-initial guess, x_0 = 0 */
  pEVSL_ParvecCopy(&b, p);
  pEVSL_ParvecNrm2(&b, &norm_r0);
  alp = 2.0 / d;
  /* x = alp * p */
  pEVSL_ParvecCopy(p, &x);
  pEVSL_ParvecScal(&x, alp);
  /* w = B * x */
  pEVSL_MatvecB(&x, w);
  /* r = b - w */
  pEVSL_ParvecCopy(&b, r);
  pEVSL_ParvecAxpy(-1.0, w, r);
  /* main iteration */
  printf("%e\n", norm_r0);
  for (i=1; i<deg; i++) {
    t = c * alp * 0.5;
    bet = t * t;
    alp = 1.0 / (d - bet);
    /* p = r + bet * p */
    pEVSL_ParvecScal(p, bet);
    pEVSL_ParvecAxpy(1.0, r, p);
    /* x = x + alp * p */
    pEVSL_ParvecAxpy(alp, p, &x);
    /* w = B * x */
    pEVSL_MatvecB(&x, w);
    /* r = b - w */
    pEVSL_ParvecCopy(&b, r);
    pEVSL_ParvecAxpy(-1.0, w, r);
  
    pEVSL_ParvecNrm2(r, &norm_r);
    printf("%e\n", norm_r);
  }
  /* result */
  pEVSL_ParvecNrm2(r, &norm_r);
  Chebdata->relres = norm_r / norm_r0;
}
