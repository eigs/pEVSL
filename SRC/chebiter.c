#include "pevsl_int.h"

//#define SAVE_CONV_HIST

/** @brief Solver function to setup Chebyshev iterations
 *
 * */
int pEVSL_ChebIterSetupMatB(int deg, int lanm, int msteps, double tol, 
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
  deg = PEVSL_MAX(deg, 0);
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

#ifdef SAVE_CONV_HIST
  PEVSL_CALLOC(data->res, deg+1, double);
#else
  data->res = NULL;
#endif

  data->comm = comm;

  return 0;
}

/** @brief Solver function of B with Chebyshev iterations
 *
 * */
void pEVSL_ChebIterSolMatBv1(double *db, double *dx, void *data) {
  int i;
  /* sizes */
  int N = pevsl_data.N;
  int n = pevsl_data.n;
  int nfirst = pevsl_data.nfirst;
  /* Cheb sol data */
  BSolDataChebiter *Chebdata = (BSolDataChebiter *) data;
  double d, c, alp=0.0, bet, t;
#ifdef SAVE_CONV_HIST
  double norm_r0, norm_r;
  double *res = Chebdata->res;
#endif
  pevsl_Parvec *w = Chebdata->w;
  pevsl_Parvec *r = Chebdata->r;
  pevsl_Parvec *p = Chebdata->p;
  /* Parvec wrapper */
  pevsl_Parvec b, x;
  int deg = Chebdata->deg;
  MPI_Comm comm = Chebdata->comm;

  /* center and half width */
  d = (Chebdata->ub + Chebdata->lb) * 0.5;
  c = (Chebdata->ub - Chebdata->lb) * 0.5;
  /* wrap b and x into pevsl_Parvec */
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &b, db);
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &x, dx);
  /* residual norm 0 */
#ifdef SAVE_CONV_HIST
  pEVSL_ParvecNrm2(&b, &norm_r0);
  res[0] = norm_r0;
#endif
  if (deg < 1) {
    return;
  }
  alp = 2.0 / d;
  /* use 0-initial guess, x_0 = 0 */
  pEVSL_ParvecCopy(&b, p);
  /* x = alp * p */
  pEVSL_ParvecCopy(p, &x);
  pEVSL_ParvecScal(&x, alp);
  /* w = B * x */
  pEVSL_MatvecB(&x, w);
  /* r = b - w */
  pEVSL_ParvecCopy(&b, r);
  pEVSL_ParvecAxpy(-1.0, w, r);
  /* main iteration */
  for (i=1; i<deg; i++) {
#ifdef SAVE_CONV_HIST
    pEVSL_ParvecNrm2(r, &norm_r);
    res[i] = norm_r;
#endif
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
  }

#ifdef SAVE_CONV_HIST
  pEVSL_ParvecNrm2(r, &norm_r);
  res[deg] = norm_r;
#endif
}

/** @brief Solver function of B with Chebyshev iterations
 * ``Iterative methods for sparse linear systems (2nd edition)'', Page 399
 * */
void pEVSL_ChebIterSolMatBv2(double *db, double *dx, void *data) {
  int i;
  /* sizes */
  int N = pevsl_data.N;
  int n = pevsl_data.n;
  int nfirst = pevsl_data.nfirst;
  /* Cheb sol data */
  BSolDataChebiter *Chebdata = (BSolDataChebiter *) data;
  double theta, delta, alpha, beta, sigma, rho, rho1;
#ifdef SAVE_CONV_HIST
  double norm_r0, norm_r;
  double *res = Chebdata->res;
#endif
  pevsl_Parvec *w = Chebdata->w;
  pevsl_Parvec *r = Chebdata->r;
  pevsl_Parvec *d = Chebdata->p;
  /* Parvec wrapper */
  pevsl_Parvec b, x;
  int deg = Chebdata->deg;
  MPI_Comm comm = Chebdata->comm;

  /* wrap b and x into pevsl_Parvec */
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &b, db);
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &x, dx);
  /* eig bounds */
  alpha = Chebdata->lb;
  beta  = Chebdata->ub;
  /* center and half width */
  theta = (beta + alpha) * 0.5;
  delta = (beta - alpha) * 0.5;
  sigma = theta / delta;
  rho   = 1.0 / sigma;
  /* use 0-initial guess, x_0 = 0, so r_0 = b */
  pEVSL_ParvecSetZero(&x);
  pEVSL_ParvecCopy(&b, r);
  /* d = 1/theta * r */
  pEVSL_ParvecCopy(r, d);
  pEVSL_ParvecScal(d, 1.0/theta);
  /* main iterations */
#ifdef SAVE_CONV_HIST
  pEVSL_ParvecNrm2(r, &norm_r0);
  res[0] = norm_r0;
#endif
  for (i=0; i<deg; i++) {
    /* x = x + d */
    pEVSL_ParvecAxpy(1.0, d, &x);
    /* w = A * d */
    pEVSL_MatvecB(d, w);
    /* r = r - w */
    pEVSL_ParvecAxpy(-1.0, w, r);
    /* rho1 = 1.0 / (2*sigma-rho) */
    rho1 = 1.0 / (2.0*sigma - rho);
    /* d = rho1*rho*d + 2*rho1/sigma*r */
    pEVSL_ParvecScal(d, rho1*rho);
    pEVSL_ParvecAxpy(2.0*rho1/delta, r, d);
    /* update rho */
    rho = rho1;
#ifdef SAVE_CONV_HIST
    pEVSL_ParvecNrm2(r, &norm_r);
    res[i+1] = norm_r;
#endif
  }
}

void pEVSL_ChebIterFree(BSolDataChebiter *data) {
  pEVSL_ParvecFree(data->w);
  pEVSL_ParvecFree(data->r);
  pEVSL_ParvecFree(data->p);
  if (data->res) {
    PEVSL_FREE(data->res);
  }
}

