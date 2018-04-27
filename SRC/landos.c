#include "pevsl_int.h"

/**
 * @file SRC/landos.c
 * @brief Function to use Lanczos method for approximating DOS for the
 * generalized eigenvalue problem.
 */

/**----------------------------------------------------------------------
 *
 *    Computes the density of states (DOS, or spectral density) using Lanczos
 *    algorithm for the generalized eigenvalue problem.
 *
 *    @param[in] pevsl pEVSL data strcut
 *    @param[in] nvec  number of sample vectors used
 *    @param[in] msteps number of Lanczos steps
 *    @param[in] npts number of sample points used for the DOS curve
 *    @param[in] intv Stores the intervals of interest
 *      intv[0:1] = [a b] = interval where DOS is to be computed
 *      intv[2:3] = [lambda_min, lambda_max] \\
 *
 *    @param[out] xdos Length-npts long vector, x-coordinate points for
 *    plotting the DOS. Must be preallocated before calling LanDos
 *
 *    @param[out] ydos Length-npts long vector, y-coordinate points for
 *    plotting the DOS. Must be preallocated before calling LanDos
 *
 *    @param[out] neig  estimated number of eigenvalues
 *    @param[in] ngroups Number of groups to partition work
 *    @param[in] groupid Group which this thread is in
 *    @param[in] gl_comm Global communicator
 *
 *
 *
 *    @note This works for both the standard and generalized eigenvalue
 *    problems.
 *    landos.c/LanDos is only for the standard eigenvalue problem.
 *----------------------------------------------------------------------*/

int pEVSL_LanDosG(pevsl_Data *pevsl, int nvec, int msteps, int npts,
                  double *xdos, double *ydos, double *neig, double* intv,
                  int ngroups, int groupid, MPI_Comm gl_comm) {

  int i, j, k;
  int maxit = msteps, m;  /* Max number of iterations */
  /*-------------------- MPI comm of this instance of pEVSL */
  MPI_Comm comm = pevsl->comm;
  /* size of the matrix */
  int N = pevsl->N;
  int n = pevsl->n;
  int nfirst = pevsl->nfirst;

  int rank;
  MPI_Comm_rank(comm, &rank);

  const int ifGenEv = pevsl->ifGenEv;

  /*-------------------- lanczos vectors updated by rotating pointer*/
  /*-------------------- pointers to Lanczos vectors */
  pevsl_Parvec parvec[6];
  pevsl_Parvec *zold  = &parvec[0];
  pevsl_Parvec *z     = &parvec[1];
  pevsl_Parvec *znew  = &parvec[2];
  pevsl_Parvec *v     = &parvec[3];
  pevsl_Parvec *vnew  = &parvec[4];
  pevsl_Parvec *vinit = &parvec[5];

  pEVSL_ParvecCreate(N, n, nfirst, comm, vinit);

  int *ind;
  PEVSL_MALLOC(ind, npts, int);
  double *y;
  PEVSL_CALLOC(y, npts, double);

  /*-------------------- frequently used constants  */
  int one = 1;
  maxit = PEVSL_MIN(N, maxit);
  size_t maxit_l = maxit;
  double *gamma2;
  PEVSL_MALLOC(gamma2, maxit, double);
  /*-----------------------------------------------------------------------*
   * *Non-restarted* Lanczos iteration
   *-----------------------------------------------------------------------
   -------------------- Lanczos vectors V_m and tridiagonal matrix T_m */
  pevsl_Parvecs *V, *Z;
  PEVSL_MALLOC(V, 1, pevsl_Parvecs);
  pEVSL_ParvecsDuplParvec(vinit, maxit+1, vinit->n_local, V);
  if (ifGenEv) {
    /* storage for Z = B * V */
    PEVSL_MALLOC(Z, 1, pevsl_Parvecs);
    pEVSL_ParvecsDuplParvec(vinit, maxit+1, vinit->n_local, Z);
  } else {
    /* Z and V are the same */
    Z = V;
  }
  /*-------------------- diag. subdiag of Tridiagional matrix */
  double *dT, *eT;
  PEVSL_MALLOC(dT, maxit, double);
  PEVSL_MALLOC(eT, maxit, double);
  double *EvalT, *EvecT;
  PEVSL_MALLOC(EvalT, maxit, double);              /* eigenvalues of tridia. matrix  T */
  PEVSL_MALLOC(EvecT, maxit_l * maxit_l, double);  /* Eigen vectors of T */

  const double lm = intv[2];
  const double lM = intv[3];
  const double aa = PEVSL_MAX(intv[0], intv[2]);
  const double bb = PEVSL_MIN(intv[1], intv[3]);
  const double kappa = 1.25;
  const int M = PEVSL_MIN(msteps, 30);
  const double H = (lM - lm) / (M - 1);
  const double sigma = H / sqrt(8 * log(kappa)) * pevsl->sigma_mult;
  const double sigma2 = 2 * sigma * sigma;
  /*-------------------- If gaussian small than tol ignore point. */
  const double tol = 1e-08;
  double width = sigma * sqrt(-2.0 * log(tol));
  linspace(aa, bb, npts, xdos);  // xdos = linspace(lm,lM, npts);

  /*-------------------- workspace [double * array] */
  double *warr;
  PEVSL_MALLOC(warr, 3*maxit, double);

  int vec_start, vec_end;
  /*-------------------- if we have more than one groups,
   *                     partition nvecs among groups */
  if (ngroups > 1) {
    pEVSL_Part1d(nvec, ngroups, &groupid, &vec_start, &vec_end, 1);
  } else {
    vec_start = 0;
    vec_end = nvec;
  }

  /*-------------------- the vector loop */
  for (m = vec_start; m < vec_end; m++) {
    /*-------------------- random vinit */
    pEVSL_ParvecRand(vinit);
    /*-------------------- a quick reference to V(:,1) */
    pEVSL_ParvecsGetParvecShell(V, 0, v);
    /*-------------------- */
    if (ifGenEv) {
      pEVSL_SolveLT(pevsl, vinit, v);
    }
    /*--------------------  normalize it */
    double t;
    if (ifGenEv) {
      /* B norm */
      pEVSL_ParvecsGetParvecShell(Z, 0, z);
      pEVSL_MatvecB(pevsl, v, z);
      pEVSL_ParvecDot(v, z, &t);
      t = 1.0 / sqrt(t);
      pEVSL_ParvecScal(z, t);
    } else {
      /* 2-norm */
      pEVSL_ParvecNrm2(vinit, &t);
      t = 1.0 / t;
      /*-------------------- copy initial vector to V(:,1) */
      pEVSL_ParvecCopy(vinit, v);
    }
    /* unit B^{-1}-norm or 2-norm */
    pEVSL_ParvecScal(v, t);
    /*-------------------- for ortho test */
    double wn = 0.0;
    int nwn = 0;
    /*--------------------  Lanczos recurrence coefficients */
    double alpha, beta = 0.0;
    /* ---------------- main Lanczos loop */
    for (k = 0; k < maxit; k++) {
      /*-------------------- quick reference to Z(:,k-1) when k>0*/
      if (k > 0) {
        pEVSL_ParvecsGetParvecShell(Z, k-1, zold);
      }
      /*-------------------- a quick reference to V(:,k) */
      pEVSL_ParvecsGetParvecShell(V, k, v);
      /*-------------------- a quick reference to Z(:,k) */
      pEVSL_ParvecsGetParvecShell(Z, k, z);
      /*-------------------- next Lanczos vector V(:,k+1)*/
      pEVSL_ParvecsGetParvecShell(V, k+1, vnew);
      /*-------------------- next Lanczos vector Z(:,k+1)*/
      pEVSL_ParvecsGetParvecShell(Z, k+1, znew);

      pEVSL_MatvecA(pevsl, v, znew);
      /*------------------ znew = znew - beta*zold */
      if (k > 0) {
        pEVSL_ParvecAxpy(-beta, zold, znew);
      }
      /*-------------------- alpha = znew'*v */
      pEVSL_ParvecDot(v, znew, &alpha);
      /*-------------------- T(k,k) = alpha */
      dT[k] = alpha;
      wn += fabs(alpha);
      /*-------------------- znew = znew - alpha*z */
      pEVSL_ParvecAxpy(-alpha, z, znew);
      /*-------------------- FULL reortho to all previous Lan vectors */
      if (ifGenEv) {
        /* znew = znew - Z(:,1:k)*V(:,1:k)'*znew */
        CGS_DGKS2(pevsl, k+1, NGS_MAX, Z, V, znew, warr);
        /* vnew = B \ znew */
        pEVSL_SolveB(pevsl, znew, vnew);
        /*-------------------- beta = (vnew, znew)^{1/2} */
        pEVSL_ParvecDot(vnew, znew, &beta);
        beta = sqrt(beta);
      } else {
        /* vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
        /* beta = norm(vnew) */
        CGS_DGKS(pevsl, k+1, NGS_MAX, V, vnew, &beta, warr);
      }
      wn += 2.0 * beta;
      nwn += 3;
      /*-------------------- lucky breakdown test */
      if (beta * nwn < orthTol * wn) {
        pEVSL_ParvecRand(vnew);
        if (ifGenEv) {
          /* znew = znew - Z(:,1:k)*V(:,1:k)'*znew */
          CGS_DGKS2(pevsl, k+1, NGS_MAX, V, Z, vnew, warr);
          /* -------------- NOTE: B-matvec */
          pEVSL_MatvecB(pevsl, vnew, znew);
          pEVSL_ParvecDot(vnew, znew, &beta);
          beta = sqrt(beta);
          /*-------------------- vnew = vnew / beta */
          t = 1.0 / beta;
          pEVSL_ParvecScal(vnew, t);
          /*-------------------- znew = znew / beta */
          pEVSL_ParvecScal(znew, t);
          beta = 0.0;
        } else {
          /* vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
          /* beta = norm(vnew) */
          CGS_DGKS(pevsl, k+1, NGS_MAX, V, vnew, &beta, warr);
          /*-------------------- vnew = vnew / beta */
          t = 1.0 / beta;
          pEVSL_ParvecScal(vnew, t);
          beta = 0.0;
        }
      } else {
        /*-------------------- vnew = vnew / beta */
        t = 1.0 / beta;
        pEVSL_ParvecScal(vnew, t);
        if (ifGenEv) {
          /*-------------------- znew = znew / beta */
          pEVSL_ParvecScal(znew, t);
        }
      }
      /*-------------------- T(k+1,k) = beta */
      eT[k] = beta;
    }

    SymmTridEig(pevsl, EvalT, EvecT, maxit, dT, eT);

    for (i = 0; i < maxit; i++) {
      /*-------------------- weights for Lanczos quadrature */
      /* Gamma2(i) = elementwise square of top entry of i-th eginvector
       * */
      gamma2[i] = EvecT[i * maxit_l] * EvecT[i * maxit_l];
    }
    /*-------------------- dos curve parameters
       Generate DOS from small gaussians centered at the ritz values */
    for (i = 0; i < maxit; i++) {
      // As msteps is width of ritzVal -> we get msteps eigenvectors
      const double t = EvalT[i];
      int numPlaced = 0;
      /*-------------------- Place elements close to t in ind */
      for (j = 0; j < npts; j++) {
        if (fabs(xdos[j] - t) < width) {
          ind[numPlaced++] = j;
        }
      }
      for (j = 0; j < numPlaced; j++) {
        y[ind[j]] += gamma2[i] *
                     exp(-((xdos[ind[j]] - t) * (xdos[ind[j]] - t)) / sigma2);
      }
    }
  } /* the vector loop */

  /* if we have more than one group */
  if (ngroups > 1) {
    double *y_global;
    PEVSL_MALLOC(y_global, npts, double);
    /* Sum of all partial results: the group leaders first do an All-reduce.
       Then, group leaders will do broadcasts
     */
    if (rank == 0) {
      MPI_Allreduce(y, y_global, npts, MPI_DOUBLE, MPI_SUM, gl_comm);
    }
    MPI_Bcast(y_global, npts, MPI_DOUBLE, 0, comm);
    memcpy(y, y_global, npts*sizeof(double));
    PEVSL_FREE(y_global);
  }

  double scaling = 1.0 / (nvec * sqrt(sigma2 * PI));
  /* y = ydos * scaling */
  DSCAL(&npts, &scaling, y, &one);
  DCOPY(&npts, y, &one, ydos, &one);
  /* input: xdos, y, output: y */
  simpson(xdos, y, npts);
  *neig = y[npts - 1] * N;

  /*-------------------- free arrays */
  pEVSL_ParvecFree(vinit);
  PEVSL_FREE(ind);
  PEVSL_FREE(y);
  PEVSL_FREE(gamma2);
  pEVSL_ParvecsFree(V);
  PEVSL_FREE(V);
  if (ifGenEv) {
    pEVSL_ParvecsFree(Z);
    PEVSL_FREE(Z);
  }
  PEVSL_FREE(dT);
  PEVSL_FREE(eT);
  PEVSL_FREE(EvalT);
  PEVSL_FREE(EvecT);
  PEVSL_FREE(warr);

  return 0;
}

/**----------------------------------------------------------------------
 *
 *    @brief Interval partitioner based for Lanczos DOS output
 *
 *    @param[in] xi coordinates of interval [a b]
 *    @param[in] yi yi[k] = integral of the does from a to xi[k]
 *    @param[in] n_int Number of desired sub-intervals
 *    @param[in] npts number of integration points (length of xi)
 *
 *    @param[out] sli Array of length n_int containing the boundaries of
 *       the intervals. [sli[i], [sli[i+1]] is the i-th interval with sli[0]
 *       = xi[0] and sli[n_int] = xi[npts-1]
 *
 *----------------------------------------------------------------------*/

void pEVSL_SpslicerLan(double* xi, double* yi, int n_int, int npts, double* sli) {
  /*-------------------- makes a call here to  integration by Simpson */
  double want;
  int k = 0;
  double t;
  int ls = 0;

  //-------------------- in-place integration ydos<--- int ydos..
  simpson(xi, yi, npts);
  //
  t = yi[0];
  want = (yi[npts - 1] - yi[0]) / (double)n_int;
  sli[ls] = xi[k];
  //-------------------- First point - t should be zero actually
  for (k = 1; k < npts; k++) {
    if (yi[k] - t >= want) {
      //-------------------- New interval defined
      ls = ls + 1;
      sli[ls] = xi[k];
      t = yi[k];
    }
  }
  //-------------------- bound for last interval is last point.
  sli[n_int] = xi[npts - 1];
}

