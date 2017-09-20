#include "pevsl_int.h"
#include "pevsl_itsol.h"

/*!
 * if want to save convergence history in Cheb iters
 */ 
#define SAVE_CONV_HIST 0

/*!
 * @brief data needed for Chebyshev iterations
 *
 */
typedef struct _Chebiter_Data {
  /* eigenvalue bounds of the matrix */
  double lb, ub;
  /* polynomial degree */
  int deg;
  /* global/local sizes and nfirst */
  int N, n, nfirst;
  /* matvec function and data */
  pevsl_Matvec *mv;
  /* work space */
  pevsl_Parvec *w, *r, *p;
  /* results */
#if SAVE_CONV_HIST
  double* res;
#endif
  /* communicator */
  MPI_Comm comm;
  /* stats */
  size_t n_chebmv;
  double t_chebmv;
} Chebiter_Data;

/** 
 * @brief Perform matrix-vector product y = A * x in Chebiter
 * 
 * */
static inline void pEVSL_ChebMatvec(Chebiter_Data   *cheb_data, 
                                    pevsl_Parvec    *x, 
                                    pevsl_Parvec    *y) {

  PEVSL_CHKERR(!cheb_data->mv);
     
  PEVSL_CHKERR(cheb_data->N != x->n_global);
  PEVSL_CHKERR(cheb_data->n != x->n_local);
  PEVSL_CHKERR(cheb_data->nfirst != x->n_first);
  PEVSL_CHKERR(cheb_data->N != y->n_global);
  PEVSL_CHKERR(cheb_data->n != y->n_local);
  PEVSL_CHKERR(cheb_data->nfirst != y->n_first);

  double tms = pEVSL_Wtime();

  cheb_data->mv->func(x->data, y->data, cheb_data->mv->data);
  
  double tme = pEVSL_Wtime();
  cheb_data->t_chebmv += tme - tms;
  cheb_data->n_chebmv ++;
}

/** @brief Return the residuals in Chebyshev iterations
 *
 * */
double* pEVSL_ChebIterGetRes(void *data) {
#if SAVE_CONV_HIST
  Chebiter_Data *cheb = (Chebiter_Data *) data;
  return (cheb->res);
#else
  return NULL;
#endif
}

/** @brief Setup Chebyshev iterations for a Parcsr matrix A for
 *         solving linear systems
 *
 * */
int pEVSL_ChebIterSetup(double lmin, double lmax, int deg, pevsl_Parcsr *A, 
                        void **data) {
  Chebiter_Data *cheb;
  PEVSL_MALLOC(cheb, 1, Chebiter_Data);
  pevsl_Parvec *w, *r, *p;
  PEVSL_MALLOC(w, 1, pevsl_Parvec);
  PEVSL_MALLOC(r, 1, pevsl_Parvec);
  PEVSL_MALLOC(p, 1, pevsl_Parvec);
  pEVSL_ParvecCreate(A->ncol_global, A->ncol_local, A->first_col, A->comm, w);
  pEVSL_ParvecDupl(w, r);
  pEVSL_ParvecDupl(w, p);

  cheb->n_chebmv = 0;
  cheb->t_chebmv = 0.0;

  /* save the solver settings */
  deg = PEVSL_MAX(deg, 0);
  cheb->lb  = lmin;
  cheb->ub  = lmax;
  cheb->deg = deg;
  cheb->N = A->ncol_global;
  cheb->n = A->ncol_local;
  cheb->nfirst = A->first_col;
  PEVSL_MALLOC(cheb->mv, 1, pevsl_Matvec);
  cheb->mv->func = pEVSL_ParcsrMatvec0;
  cheb->mv->data = (void *) A;
  /* alloc work space */
  cheb->w = w;
  cheb->r = r;
  cheb->p = p;

#if SAVE_CONV_HIST
  PEVSL_CALLOC(cheb->res, deg+1, double);
#endif
  cheb->comm = A->comm;
  *data = (void *) cheb;

  return 0;
}

/** @brief Solve function for Chebyshev iterations [Version 1]
 * ``Templates for the Solution of Algebraic Eigenvalue Problems: 
     a Practical Guide''
 */
void pEVSL_ChebIterSolv1(double *db, double *dx, void *data) {
  int i;
  /* Cheb sol data */
  Chebiter_Data *Chebdata = (Chebiter_Data *) data;
  double d, c, alp=0.0, bet, t;
#if SAVE_CONV_HIST
  double norm_r0, norm_r;
  double *res = Chebdata->res;
#endif
  /* sizes and nfirst */
  int N = Chebdata->N;
  int n = Chebdata->n;
  int nfirst = Chebdata->nfirst;

  pevsl_Parvec *w = Chebdata->w;
  pevsl_Parvec *r = Chebdata->r;
  pevsl_Parvec *p = Chebdata->p;
  /* Parvec wrapper */
  pevsl_Parvec b, x;
  int deg = Chebdata->deg;
  MPI_Comm comm = Chebdata->comm;
  /* matvec */
  //pevsl_Matvec *mv = Chebdata->mv;

  /* center and half width */
  d = (Chebdata->ub + Chebdata->lb) * 0.5;
  c = (Chebdata->ub - Chebdata->lb) * 0.5;
  /* wrap b and x into pevsl_Parvec */
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &b, db);
  pEVSL_ParvecCreateShell(N, n, nfirst, comm, &x, dx);
  /* residual norm 0 */
#if SAVE_CONV_HIST
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
  /* w = C * x */
  pEVSL_ChebMatvec(Chebdata, &x, w);
  //mv->func(x.data, w->data, mv->data);
  /* r = b - w */
  pEVSL_ParvecCopy(&b, r);
  pEVSL_ParvecAxpy(-1.0, w, r);
  /* main iteration */
  for (i=1; i<deg; i++) {
#if SAVE_CONV_HIST
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
    /* w = C * x */
    pEVSL_ChebMatvec(Chebdata, &x, w);
    //mv->func(x.data, w->data, mv->data);
    /* r = b - w */
    pEVSL_ParvecCopy(&b, r);
    pEVSL_ParvecAxpy(-1.0, w, r);
  }

#if SAVE_CONV_HIST
  pEVSL_ParvecNrm2(r, &norm_r);
  res[deg] = norm_r;
#endif
}


/** @brief Solve function for Chebyshev iterations [Version 2]
 * Y. Saad, ``Iterative methods for sparse linear systems (2nd edition)'', 
 * Page 399
 */
void pEVSL_ChebIterSolv2(double *db, double *dx, void *data) {
  int i;
  /* Cheb sol data */
  Chebiter_Data *Chebdata = (Chebiter_Data *) data;
  double theta, delta, alpha, beta, sigma, rho, rho1;
#if SAVE_CONV_HIST
  double norm_r0, norm_r;
  double *res = Chebdata->res;
#endif
  /* sizes and nfirst */
  int N = Chebdata->N;
  int n = Chebdata->n;
  int nfirst = Chebdata->nfirst;
  
  pevsl_Parvec *w = Chebdata->w;
  pevsl_Parvec *r = Chebdata->r;
  pevsl_Parvec *d = Chebdata->p;
  /* Parvec wrapper */
  pevsl_Parvec b, x;
  int deg = Chebdata->deg;
  MPI_Comm comm = Chebdata->comm;
  /* matvec */
  //pevsl_Matvec *mv = Chebdata->mv;

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
#if SAVE_CONV_HIST
  pEVSL_ParvecNrm2(r, &norm_r0);
  res[0] = norm_r0;
#endif
  for (i=0; i<deg; i++) {
    /* x = x + d */
    pEVSL_ParvecAxpy(1.0, d, &x);
    /* w = C * d */
    //mv->func(d->data, w->data, mv->data);
    pEVSL_ChebMatvec(Chebdata, d, w);
    /* r = r - w */
    pEVSL_ParvecAxpy(-1.0, w, r);
    /* rho1 = 1.0 / (2*sigma-rho) */
    rho1 = 1.0 / (2.0*sigma - rho);
    /* d = rho1*rho*d + 2*rho1/sigma*r */
    pEVSL_ParvecScal(d, rho1*rho);
    pEVSL_ParvecAxpy(2.0*rho1/delta, r, d);
    /* update rho */
    rho = rho1;
#if SAVE_CONV_HIST
    pEVSL_ParvecNrm2(r, &norm_r);
    res[i+1] = norm_r;
#endif
  }
}

void pEVSL_ChebIterFree(void *vdata) {
  Chebiter_Data *data = (Chebiter_Data *) vdata;
  pEVSL_ParvecFree(data->w);
  pEVSL_ParvecFree(data->r);
  pEVSL_ParvecFree(data->p);
  PEVSL_FREE(data->w);
  PEVSL_FREE(data->r);
  PEVSL_FREE(data->p);
  PEVSL_FREE(data->mv);
#if SAVE_CONV_HIST
  PEVSL_FREE(data->res);
#endif
  PEVSL_FREE(vdata);
}

void pEVSL_ChebIterStatsPrint(void *data, FILE *fstats) {
  
  Chebiter_Data *cheb = (Chebiter_Data *) data;
  
  MPI_Comm comm = cheb->comm;
  double t_chebmv;
  unsigned long n_chebmv;
  int rank;
  
  /* rank 0 prints */
  MPI_Comm_rank(comm, &rank);
  
  MPI_Reduce(&cheb->t_chebmv,     &t_chebmv,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
  n_chebmv = cheb->n_chebmv;

  if (rank == 0) {
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
    if (n_chebmv)  { fprintf(fstats, "   Matvec in ChebIter        :  %f (%8ld, avg %f)\n",  t_chebmv, n_chebmv, t_chebmv / n_chebmv); }
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
  }
}

