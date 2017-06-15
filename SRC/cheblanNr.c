#include "pevsl_protos.h"

/**
 * if filter the initial vector
 */ 
#define FILTER_VINIT 1

/**-----------------------------------------------------------------------
 *  @brief Chebyshev polynomial filtering Lanczos process [NON-restarted version]
 *
 *  @param intv     An array of length 4 \n
 *          [intv[0], intv[1]] is the interval of desired eigenvalues \n
 *          [intv[2], intv[3]] is the global interval of all eigenvalues \n
 *          Must contain all eigenvalues of A
 *  
 *  @param maxit    Max number of outer Lanczos steps  allowed --[max dim of Krylov 
 *          subspace]
 *  
 *  @param tol       
 *          Tolerance for convergence. The code uses a stopping criterion based
 *          on the convergence of the restricted trace. i.e., the sum of the
 *          eigenvalues of T_k that  are in the desired interval. This test  is
 *          rather simple since these eigenvalues are between `bar' and  1.0.
 *          We want the relative error on  this restricted  trace to be  less
 *          than  tol.  Note that the test  performed on filtered matrix only
 *          - *but* the actual residual norm associated with the original
 *          matrix A is returned
 *  
 *  @param vinit  initial  vector for Lanczos -- [optional]
 * 
 *  @param pol       A struct containing the parameters of the polynomial..
 *  This is set up by a call to find_deg prior to calling chenlanNr 
 * 
 *  @b Modifies:
 *  @param[out] nevOut    Number of eigenvalues/vectors computed
 *  @param[out] Wo        A set of eigenvectors  [n x nevOut matrix]
 *  @param[out] reso      Associated residual norms [nev x 1 vector]
 *  @param[out] lamo      Lambda computed
 *  @param[out] fstats    File stream which stats are printed to
 *
 *  @return Returns 0 on success (or if check_intv() is non-positive),  -1
 *  if |gamB| < 1
 *
 *
 * @warning memory allocation for Wo/lamo/reso within this function 
 * ------------------------------------------------------------ */
int pEVSL_ChebLanNr(double *intv, int maxit, double tol, pevsl_Parvec *vinit, 
                    pevsl_Polparams *pol, int *nevOut, double **lamo, pevsl_Parvec **Wo, 
                    double **reso, MPI_Comm comm, FILE *fstats) {
  /*-------------------- for stats */
  double tms = pEVSL_Wtime();
  double tr0, tr1;
  double *y, flami; 
  int i, j, k, kdim=0, rank;
  /* handle case where fstats is NULL. Then no output. Needed for openMP */
  int do_print = 1;   
  if (fstats == NULL) {
    do_print = 0;
  }
  /*-------------------- frequently used constants  */
  //char cN = 'N';   
  //int one = 1;
  //double done=1.0,dzero=0.0;
  /* MPI communicator */
  /*
  MPI_Comm_compare(comm, vinit->comm, &comp);
  if (comp != MPI_IDENT) {
    return 1;
  }
  */
  /*-------------------- rank in comm */
  MPI_Comm_rank(comm, &rank);
  /*-------------------- Ntest = when to start testing convergence */
  int Ntest = 30; 
  /*-------------------- how often to test */
  int cycle = 20; 
  /* size of the matrix. N: global size */
  int N;
  N = pevsl_data.N;
  /*-------------------- the communicator working on */
  maxit = PEVSL_MIN(N, maxit);
  /*-------------------- polynomial filter  approximates the delta
                         function centered at 'gamB'. 
                         bar: a bar value to threshold Ritz values of p(A) */
  double bar = pol->bar;
  double gamB = pol->gam;
  /*-------------------- interval [aa, bb], used for testing only at end */
  if (check_intv(intv, fstats) < 0) {
    *nevOut = 0;
    *lamo = NULL; *Wo = NULL; *reso = NULL;
    return 0;
  }
  double aa = intv[0];
  double bb = intv[1];
  int deg = pol->deg;
  if (do_print) {
    pEVSL_fprintf0(rank, fstats, "intv:[%e, %e, %e, %e]\n", intv[0], intv[1], intv[2], intv[3]);
    pEVSL_fprintf0(rank, fstats, " ** Cheb Poly of deg = %d, gam = %.15e, bar: %.15e\n", 
                  deg, gamB, bar);
  }
  /*-------------------- gamB must be within [-1, 1] */
  if (gamB > 1.0 || gamB < -1.0) {
    pEVSL_fprintf0(rank, stdout, "gamB error %.15e\n", gamB);
    return -1;
  }
  /*-----------------------------------------------------------------------* 
   * *Non-restarted* Lanczos iteration 
   *-----------------------------------------------------------------------*/
  if (do_print) {
    pEVSL_fprintf0(rank, fstats, " ** Cheb-LanNr \n");
  }
  /*-------------------- Lanczos vectors V_m and tridiagonal matrix T_m */
  pevsl_Parvec *V, *Rvec, *Z;
  double *dT, *eT;
  PEVSL_MALLOC(V, maxit+1, pevsl_Parvec);
  for (i=0; i<maxit+1; i++) {
    pEVSL_ParvecDupl(vinit, &V[i]);
  }
  if (pevsl_data.ifGenEv) {
    /* storage for Z = B * V */
    PEVSL_MALLOC(Z, maxit+1, pevsl_Parvec);
    for (i=0; i<maxit+1; i++) {
      pEVSL_ParvecDupl(vinit, &Z[i]);
    }
  } else {
    /* Z and V are the same */
    Z = V;
  }
  /*-------------------- diag. subdiag of Tridiagional matrix */
  PEVSL_MALLOC(dT, maxit, double);
  PEVSL_MALLOC(eT, maxit, double);
  double *Lam, *res, *EvalT, *EvecT;
  /*-------------------- Lam, Rvec: the converged (locked) Ritz values vecs*/
  PEVSL_MALLOC(Lam, maxit, double);         // holds computed Ritz values
  PEVSL_MALLOC(res, maxit, double);         // residual norms (w.r.t. ro(A))
  PEVSL_MALLOC(EvalT, maxit, double);       // eigenvalues of tridia. matrix  T
  //PEVSL_MALLOC(EvecT, maxit*maxit, double); // Eigen vectors of T
  /*-------------------- nev = current number of converged e-pairs 
                         nconv = converged eigenpairs from looking at Tk alone */
  int nev, nconv = 0;
  /*-------------------- nmv counts  matvecs */
  //int nmv = 0;
  /*-------------------- u  is just a pointer. wk == work space */
  pevsl_Parvec *wk, *w2, *vrand = NULL, *u;
  int wk_size = pevsl_data.ifGenEv ? 4 : 3;
  PEVSL_MALLOC(wk, wk_size, pevsl_Parvec);
  for (i=0; i<wk_size; i++) {
    pEVSL_ParvecDupl(vinit, &wk[i]);
  }
  w2 = wk + 1;
#if FILTER_VINIT
  /*-------------------- compute w = p[(A-cc)/dd] * v */
  /*------------------  Filter the initial vector*/
  pEVSL_ChebAv(pol, vinit, V, wk);    
  PEVSL_MALLOC(vrand, 1, pevsl_Parvec);
  pEVSL_ParvecDupl(vinit, vrand);
#else
  /*-------------------- copy initial vector to V(:,1) */
  pEVSL_ParvecCopy(vinit, V);
#endif
  /*--------------------  normalize it */
  double t, nt, res0;
  if (pevsl_data.ifGenEv) {
    /* B norm */
    pEVSL_MatvecB(V, Z);
    pEVSL_ParvecDot(V, Z, &t);
    t = 1.0 / sqrt(t);
    pEVSL_ParvecScal(Z, t);
  } else {
    /* 2-norm */
    pEVSL_ParvecNrm2(V, &t);
    t = 1.0 / t;
  }
  /* unit B-norm or 2-norm */
  pEVSL_ParvecScal(V, t);
  /*-------------------- for ortho test */
  double wn = 0.0;
  int nwn = 0;
  /*-------------------- for stopping test [restricted trace]*/
  tr0 = 0;
  /*-------------------- lanczos vectors updated by rotating pointer*/
  /*-------------------- pointers to Lanczos vectors */
  pevsl_Parvec *zold, *z, *znew, *v, *vnew;
  /*--------------------  Lanczos recurrence coefficients */
  double alpha, beta=0.0;
  int count = 0;
  // ---------------- main Lanczos loop
  for (k=0; k<maxit; k++) {
    /*-------------------- quick reference to Z(:,k-1) when k>0*/
    zold = k > 0 ? Z+k-1 : NULL;
    /*-------------------- a quick reference to V(:,k) */
    v = &V[k];
    /*-------------------- a quick reference to Z(:,k) */
    z = &Z[k];
    /*-------------------- next Lanczos vector V(:,k+1)*/
    vnew = v + 1;
    /*-------------------- next Lanczos vector Z(:,k+1)*/
    znew = z + 1;
    /*-------------------- compute w = p[(A-cc)/dd] * v */
    /*------------------ NOTE: z is used!!! [TODO: FIX ME] */
    pEVSL_ChebAv(pol, z, znew, wk);
    /*------------------ znew = znew - beta*zold */
    if (zold) {
      //nbeta = -beta;
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
    if (pevsl_data.ifGenEv) {
      /* znew = znew - Z(:,1:k)*V(:,1:k)'*znew */
      MGS_DGKS2(k+1, NGS_MAX, Z, V, znew);
      /* vnew = B \ znew */
      pEVSL_SolveB(znew, vnew);
      /*-------------------- beta = (vnew, znew)^{1/2} */
      pEVSL_ParvecDot(vnew, znew, &beta);
      beta = sqrt(beta);
    } else {
      /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
      /*   beta = norm(vnew) */
      MGS_DGKS(k+1, NGS_MAX, V, vnew, &beta);
    }
    wn += 2.0 * beta;
    nwn += 3;
    /*-------------------- lucky breakdown test */
    if (beta*nwn < orthTol*wn) {
      if (do_print) {
        pEVSL_fprintf0(rank, fstats, "it %4d: Lucky breakdown, beta = %.15e\n", k, beta);
      }
#if FILTER_VINIT
      /*------------------ generate a new init vector */
      pEVSL_ParvecRand(vrand);
      /*------------------  Filter the initial vector*/
      pEVSL_ChebAv(pol, vrand, vnew, wk);
#else
      pEVSL_ParvecRand(vnew);
#endif
      if (pevsl_data.ifGenEv) {
      /* vnew = vnew - V(:,1:k)*Z(:,1:k)'*vnew */
        MGS_DGKS2(k+1, NGS_MAX, V, Z, vnew);
        pEVSL_MatvecB(vnew, znew);
        pEVSL_ParvecDot(vnew, znew, &beta);
        beta = sqrt(beta); 
        t = 1.0 / beta;
        pEVSL_ParvecScal(vnew, t);
        pEVSL_ParvecScal(znew, t);
        beta = 0.0;
      } else {
        MGS_DGKS(k+1, NGS_MAX, V, vnew, &beta);
        t = 1.0 / beta;
        pEVSL_ParvecScal(vnew, t);
        beta = 0.0;      
      }
    } else {
      /*-------------------- vnew = vnew / beta */
      t = 1.0 / beta;
      pEVSL_ParvecScal(vnew, t);
      if (pevsl_data.ifGenEv) {
        /*-------------------- znew = znew / beta */
        pEVSL_ParvecScal(znew, t);
      }
    }
    /*-------------------- T(k+1,k) = alpha */
    eT[k] = beta;
    
    /*---------------------- test for Ritz vectors */
    if ( (k < Ntest || (k-Ntest) % cycle != 0) && k != maxit-1 ) {
      continue;
    }
    /*---------------------- diagonalize  T(1:k,1:k)       
                             vals in EvalT, vecs in EvecT  */
    kdim = k+1;
#if 1
    /*-------------------- THIS uses dsetv, do not need eig vector */    
    SymmTridEig(EvalT, NULL, kdim, dT, eT);
    count = kdim;
#else
    /*-------------------- THIS uses dstemr */
    double vl = bar - DBL_EPSILON, vu = 2.0;  /* needed by SymmTridEigS */
    SymmTridEigS(EvalT, EvecT, kdim, vl, vu, &count, dT, eT);
#endif
    /*-------------------- restricted trace: used for convergence test */
    tr1 = 0;  
    /*-------------------- get residual norms and check acceptance of Ritz 
     *                     values for p(A). nconv records number of 
     *                     eigenvalues whose residual for p(A) is smaller 
     *                     than tol. */
    nconv = 0;
    for (i=0; i<count; i++) {
      flami = EvalT[i];
      if (flami + DBL_EPSILON >= bar) {
        tr1+= flami;
        nconv++;
      }
    }

    if (do_print) {
      pEVSL_fprintf0(rank, fstats, "k %4d:   nconv %4d  tr1 %21.15e\n",
                     k, nconv,tr1);
    }
    /* -------------------- simple test because all eigenvalues
                            are between gamB and ~1. */
    if (fabs(tr1-tr0) < tol*fabs(tr1)) {
      break;
    }
    tr0 = tr1;
  } /* end of the main loop */

  /*-------------------- compute eig vals and vector */    
  PEVSL_MALLOC(EvecT, kdim*kdim, double); // Eigen vectors of T
  SymmTridEig(EvalT, EvecT, kdim, dT, eT);
  
  /*-------------------- done == compute Ritz vectors */
  PEVSL_MALLOC(Rvec, nconv, pevsl_Parvec);
  for (i=0; i<nconv; i++) {
    pEVSL_ParvecDupl(vinit, &Rvec[i]); // holds computed Ritz vectors
  }
  nev = 0;
  for (i=0; i<count; i++) {
    flami = EvalT[i];
    //-------------------- reject eigenvalue if rho(lam)<bar
    if (flami < bar) {
      continue;
    }
    y = &EvecT[i*kdim];
    /*-------------------- make sure to normalize */
    /*
    t = DNRM2(&kdim, y, &one);
    t = 1.0 / t;
    DSCAL(&kdim, &t, y, &one);
    */
    /*-------------------- compute Ritz vectors */
    u = &Rvec[nev];
    pEVSL_ParvecSetZero(u);
    for (j=0; j<kdim; j++) {
      pEVSL_ParvecAxpy(y[j], V+j, u);
    }
    /*-------------------- normalize u */
    if (pevsl_data.ifGenEv) {
      /* B-norm, w2 = B*u */
      pEVSL_MatvecB(u, w2);
      pEVSL_ParvecDot(u, w2, &t);
      t = sqrt(t); /* should be one */
    } else {
      /* 2-norm */
      //t = DNRM2(&n, u, &one); 
      pEVSL_ParvecNrm2(u, &t); /* should be one */
    }
    /*-------------------- return code 2 --> zero eigenvector found */
    if (t == 0.0) {
      return 2;
    }
    /*-------------------- scal u */
    t = 1.0 / t;
    pEVSL_ParvecScal(u, t);
    if (pevsl_data.ifGenEv) {
      /*------------------ w2 = B*u */
      pEVSL_ParvecScal(w2, t);
    }
    /*-------------------- w = A*u */
    pEVSL_MatvecA(u, wk);
    /*-------------------- Ritz val: t = (u'*w)/(u'*u)
                                     t = (u'*w)/(u'*B*u) */
    pEVSL_ParvecDot(wk, u, &t);
    /*-------------------- if lambda (==t) is in [a,b] */
    if (t < aa - DBL_EPSILON || t > bb + DBL_EPSILON) {
      continue;
    }
    /*-------------------- compute residual wrt A for this pair */
    nt = -t;
    if (pevsl_data.ifGenEv) {
      /*-------------------- w = w - t*B*u */
      pEVSL_ParvecAxpy(nt, w2, wk);
      /*-------------------- res0 = norm(w) */
      pEVSL_ParvecNrm2(wk, &res0); 
    } else {
      /*-------------------- w = w - t*u */
      pEVSL_ParvecAxpy(nt, u, wk);
      /*-------------------- res0 = norm(w) */
      pEVSL_ParvecNrm2(wk, &res0); 
    }
    /*--------------------   accept (t, y) */
    if (res0 < tol) {
      Lam[nev] = t;
      res[nev] = res0;
      nev++;
    }
  }

  /*-------------------- Done.  output : */
  *nevOut = nev;
  *lamo = Lam;
  *Wo = Rvec;
  *reso = res;
  /*-------------------- free arrays */
  for (i=0; i<maxit+1; i++) {
    pEVSL_ParvecFree(&V[i]);
  }
  PEVSL_FREE(V);
  PEVSL_FREE(dT);
  PEVSL_FREE(eT);
  PEVSL_FREE(EvalT);
  PEVSL_FREE(EvecT);
  for (i=0; i<wk_size; i++) {
    pEVSL_ParvecFree(&wk[i]);
  }
  PEVSL_FREE(wk);
  if (vrand) {
    pEVSL_ParvecFree(vrand);
    PEVSL_FREE(vrand);
  }
  if (pevsl_data.ifGenEv) {
    for (i=0; i<maxit+1; i++) {
      pEVSL_ParvecFree(&Z[i]);
    }
    PEVSL_FREE(Z);
  }

  double tme = pEVSL_Wtime();
  pevsl_stat.t_solver += tme - tms;

  return 0;
}


