#include "pevsl_int.h"

#define COMP_RES 0

/**
 * @file lanTrbounds.c
 * @brief A more robust algorithm to give bounds of spectrum based on TR
 * Lanczos
 */
/**
 * @brief Lanczos process for eigenvalue bounds [Thick restart version]
 *
 * @param[in] lanm      Dimension of Krylov subspace [restart dimension]
 * 
 * @param[in] maxit  max Num of outer Lanczos iterations (restarts) allowed -- 
 *         Each restart may or use the full lanm lanczos steps or fewer.
 * 
 * @param[in] tol       tolerance for convergence
 * @param[in] vinit     initial  vector for Lanczos -- [optional]
 * @param[in] bndtype   Type of bound >1 for kato-temple, otherwise
 *                      simple
 *
 * @param[out] lammin   Lower bound of the spectrum
 * @param[out] lammax   Upper bound of the spectrum
 * @param[out] fstats File stream which stats are printed to
 *
 * @return Returns 0 on success 
 *
 **/
int pEVSL_LanTrbounds(pevsl_Data *pevsl, int lanm, int maxit, double tol, 
                      pevsl_Parvec *vinit, int bndtype,
                      double *lammin, double *lammax, FILE *fstats) {

  double tms = pEVSL_Wtime();
  const int ifGenEv = pevsl->ifGenEv;
  double lmin=0.0, lmax=0.0, t, t1, t2;
  int do_print = 1, rank;
  /* handle case where fstats is NULL. Then no output. Needed for openMP. */
  if (fstats == NULL) {
    do_print = 0;
  }
  MPI_Comm comm = pevsl->comm; 
  /*-------------------- MPI rank in comm */
  MPI_Comm_rank(comm, &rank);  
  /* size of the matrix */
  int N;
  N = pevsl->N;
  /*--------------------- adjust lanm and maxit */
  lanm = PEVSL_MIN(lanm, N);
  int lanm1=lanm+1;
  /*  if use full lanczos, should not do more than n iterations */
  if (lanm == N) {
    maxit = PEVSL_MIN(maxit, N);
  }
  size_t lanm1_l = lanm1;
  /*--------------------   some constants frequently used */
  int i;
  /*-----------------------------------------------------------------------* 
   * *thick restarted* Lanczos step 
   *-----------------------------------------------------------------------*/
  if (do_print) {
    fprintf(fstats, " LanTR for bounds: dim %d, maxits %d\n", lanm, maxit);
  }
  /*--------------------- the min number of steps to be performed for 
   *                      each innter loop, must be >= 1 - if not, 
   *                      it means that the Krylov dim is too small */
  int min_inner_step = 5;
  /*-------------------- it = number of Lanczos steps */
  int it = 0;
  /*-------------------- Lanczos vectors V_m and tridiagonal matrix T_m */
  pevsl_Parvecs *V, *Z;
  double *T;
  PEVSL_MALLOC(V, 1, pevsl_Parvecs);
  pEVSL_ParvecsDuplParvec(vinit, lanm1, vinit->n_local, V);
  /*-------------------- for gen eig prob, storage for Z = B * V */
  if (ifGenEv) {
    PEVSL_MALLOC(Z, 1, pevsl_Parvecs);
    pEVSL_ParvecsDuplParvec(vinit, lanm1, vinit->n_local, Z);
  } else {
    Z = V;
  }
  /*-------------------- T must be zeroed out initially */
  PEVSL_CALLOC(T, lanm1_l*lanm1_l, double);
  /*-------------------- trlen = dim. of thick restart set */
  int trlen = 0;
  /*-------------------- Ritz values and vectors of p(A) */
  double *Rval;
  PEVSL_MALLOC(Rval, lanm, double);
  /*-------------------- Only compute 2 Ritz vectors */
  pevsl_Parvec Rvec[2], BRvec[2];
  pEVSL_ParvecDupl(vinit, &Rvec[0]);
  pEVSL_ParvecDupl(vinit, &Rvec[1]);
  if (ifGenEv) {
    pEVSL_ParvecDupl(vinit, &BRvec[0]);
    pEVSL_ParvecDupl(vinit, &BRvec[1]);
  }
  /*-------------------- Eigen vectors of T */
  double *EvecT;
  PEVSL_MALLOC(EvecT, lanm1_l*lanm1_l, double);
  /*-------------------- s used by TR (the ``spike'' of 1st block in Tm)*/
  double s[3];
  /*-------------------- alloc some work space */
  double *warr;
  PEVSL_MALLOC(warr, 3*lanm, double);  
  /*-------------------- copy initial vector to V(:,1) */
  pevsl_Parvec parvec[5];
  pevsl_Parvec *v    = &parvec[0];
  pevsl_Parvec *vnew = &parvec[1];
  pevsl_Parvec *z    = &parvec[2];
  pevsl_Parvec *znew = &parvec[3];
  pevsl_Parvec *zold = &parvec[4];

  /* v references the 1st columns of V */
  pEVSL_ParvecsGetParvecShell(V, 0, v);
  pEVSL_ParvecCopy(vinit, v);
  /*-------------------- normalize it */
  if (ifGenEv) {
    /* z references the 1st columns of Z */
    pEVSL_ParvecsGetParvecShell(Z, 0, z);
    /* B norm */
    pEVSL_MatvecB(pevsl, v, z);
    pEVSL_ParvecDot(v, z, &t);
    t = 1.0 / sqrt(t);
    /* z = B*v */
    pEVSL_ParvecScal(z, t);
  } else {
    /* 2-norm */
    pEVSL_ParvecNrm2(v, &t);
    t = 1.0 / t;
  }
  /* unit B-norm or 2-norm */
  pEVSL_ParvecScal(v, t);
  /*-------------------- main (restarted Lan) outer loop */
  while (it < maxit) {
    /*-------------------- for ortho test */
    double wn = 0.0;
    int nwn = 0;
    /*  beta */
    double beta = 0.0;
    /*  start with V(:,k) */
    int k = trlen > 0 ? trlen + 1 : 0;
    /* ! add a test if dimension exceeds (m+1) 
     * (trlen + 1) + min_inner_step <= lanm + 1 */
    if (k+min_inner_step > lanm1) {
      pEVSL_fprintf0(rank, stderr, "Krylov dim too small for this problem. Try a larger dim\n");
      exit(1);
    }
    /*-------------------- thick restart special step */
    if (trlen > 0) {
      int k1 = k-1;
      /*------------------ a quick reference to V(:,k) */
      pEVSL_ParvecsGetParvecShell(V, k1, v);
      pEVSL_ParvecsGetParvecShell(Z, k1, z);
      /*------------------ next Lanczos vector */
      pEVSL_ParvecsGetParvecShell(V, k1+1, vnew);
      pEVSL_ParvecsGetParvecShell(Z, k1+1, znew);
      /*------------------ znew = A * v */
      pEVSL_MatvecA(pevsl, v, znew);
      /*-------------------- restart with 'trlen' Ritz values/vectors
                             T = diag(Rval(1:trlen)) */
      for (i=0; i<trlen; i++) {
        T[i*lanm1_l+i] = Rval[i];
        wn += fabs(Rval[i]);
      }
      /*--------------------- s(k) = V(:,k)'* znew */
      pEVSL_ParvecDot(v, znew, s+k1);
      /*--------------------- znew = znew - Z(:,1:k)*s(1:k) */
      pEVSL_ParvecsGemv(-1.0, Z, k, s, 1.0, znew);
      /*-------------------- expand T matrix to k-by-k, arrow-head shape
                             T = [T, s(1:k-1)] then T = [T; s(1:k)'] */
      for (i=0; i<k1; i++) {
        T[trlen*lanm1_l+i] = s[i];
        T[i*lanm1_l+trlen] = s[i];
        wn += 2.0 * fabs(s[i]);
      }
      T[trlen*lanm1_l+trlen] = s[k1];
      wn += fabs(s[k1]);
      if (ifGenEv) {
        /*-------------------- vnew = B \ znew */
        pEVSL_SolveB(pevsl, znew, vnew);
        /*-------------------- beta = (vnew, znew)^{1/2} */
        pEVSL_ParvecDot(vnew, znew, &beta);
        beta = sqrt(beta);
      } else {
        /*-------------------- beta = norm(w) */
        pEVSL_ParvecNrm2(vnew, &beta);
      }
      wn += 2.0 * beta;
      nwn += 3*k;
      /*   beta ~ 0 */
      if (beta*nwn < orthTol*wn) {
        pEVSL_ParvecRand(vnew);
        if (ifGenEv) {
          /* vnew = vnew - V(:,1:k)*Z(:,1:k)'*vnew */
          CGS_DGKS2(pevsl, k, NGS_MAX, V, Z, vnew, warr);          
          pEVSL_MatvecB(pevsl, vnew, znew);
          pEVSL_ParvecDot(vnew, znew, &beta);
          beta = sqrt(beta);         
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          pEVSL_ParvecScal(znew, ibeta);
          beta = 0.0;            
        } else {
          /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
          /*   beta = norm(w) */
          CGS_DGKS(pevsl, k, NGS_MAX, V, vnew, &beta, warr);
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          beta = 0.0;
        }
      } else {
        /*------------------- w = w / beta */
        double ibeta = 1.0 / beta;
        pEVSL_ParvecScal(vnew, ibeta);
        if (ifGenEv) {
          pEVSL_ParvecScal(znew, ibeta);
        }
      }
      /*------------------- T(k+1,k) = beta; T(k,k+1) = beta; */
      T[k1*lanm1_l+k] = beta;
      T[k*lanm1_l+k1] = beta;
    } /* if (trlen > 0) */
    /*-------------------- Done with TR step. Rest of Lanczos step */
    /*-------------------- regardless of trlen, *(k+1)* is the current 
     *                     number of Lanczos vectors in V */
    /*-------------------- pointer to the previous Lanczos vector */
    if (k > 0) {
      pEVSL_ParvecsGetParvecShell(Z, k-1, zold);
    } else {
      pEVSL_ParvecsGetParvecShell(Z, -1, zold); /* zold->data = NULL; */
    }
    /*------------------------------------------------------*/
    /*------------------ Lanczos inner loop ----------------*/
    /*------------------------------------------------------*/
    while (k < lanm && it < maxit) {
      k++;
      /*---------------- a quick reference to V(:,k) */
      pEVSL_ParvecsGetParvecShell(V, k-1, v);
      pEVSL_ParvecsGetParvecShell(Z, k-1, z);
      /*---------------- next Lanczos vector */
      pEVSL_ParvecsGetParvecShell(V, k, vnew);
      pEVSL_ParvecsGetParvecShell(Z, k, znew);
      /*------------------ znew = A * v */
      pEVSL_MatvecA(pevsl, v, znew);
      it++;
      /*-------------------- znew = znew - beta*zold */
      if (zold->data) {
        pEVSL_ParvecAxpy(-beta, zold, znew);
      }
      /*-------------------- alpha = znew'*v */
      double alpha;
      pEVSL_ParvecDot(v, znew, &alpha);
      /*-------------------- T(k,k) = alpha */
      T[(k-1)*lanm1_l+(k-1)] = alpha;
      wn += fabs(alpha);
      /*-------------------- znew = znew - alpha*z */
      pEVSL_ParvecAxpy(-alpha, z, znew);
      /*-------------------- FULL reortho to all previous Lan vectors */
      if (ifGenEv) {
        /* znew = znew - Z(:,1:k)*V(:,1:k)'*znew */
        CGS_DGKS2(pevsl, k, NGS_MAX, Z, V, znew, warr);
        /* vnew = B \ znew */
        pEVSL_SolveB(pevsl, znew, vnew);
        /*-------------------- beta = (vnew, znew)^{1/2} */
        pEVSL_ParvecDot(vnew, znew, &beta);
        beta = sqrt(beta);
      } else {
        /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
        /*   beta = norm(w) */
        CGS_DGKS(pevsl, k, NGS_MAX, V, vnew, &beta, warr);
      }
      wn += 2.0 * beta;
      nwn += 3;
      /*-------------------- zold = z */
      zold->data = z->data;
      /*-------------------- lucky breakdown test */
      if (beta*nwn < orthTol*wn) {
        if (do_print) {
          pEVSL_fprintf0(rank, fstats, "it %4d: Lucky breakdown, beta = %.15e\n", it, beta);
        }
        /* generate a new init vector */
        pEVSL_ParvecRand(vnew);
        if (ifGenEv) {
          /* vnew = vnew - V(:,1:k)*Z(:,1:k)'*vnew */
          CGS_DGKS2(pevsl, k, NGS_MAX, V, Z, vnew, warr);          
          pEVSL_MatvecB(pevsl, vnew, znew);
          pEVSL_ParvecDot(vnew, znew, &beta);
          beta = sqrt(beta); 
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          pEVSL_ParvecScal(znew, ibeta);
          beta = 0.0;            
        } else {
          /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
          /*   beta = norm(w) */
          CGS_DGKS(pevsl, k, NGS_MAX, V, vnew, &beta, warr);          
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          beta = 0.0;
        }
      } else {
        /*---------------------- vnew = vnew / beta */
        double ibeta = 1.0 / beta;
        pEVSL_ParvecScal(vnew, ibeta);
        if (ifGenEv) {
          /*-------------------- znew = znew / beta */
          pEVSL_ParvecScal(znew, ibeta);
        }
      }
      /*-------------------- T(k,k+1) = T(k+1,k) = beta */
      T[k*lanm1_l+(k-1)] = beta;
      T[(k-1)*lanm1_l+k] = beta;
    } /* while (k<mlan) loop */

    /*-------------------- solve eigen-problem for T(1:k,1:k)
                           vals in Rval, vecs in EvecT */
    SymEigenSolver(pevsl, k, T, lanm1, EvecT, lanm1, Rval);

    /*-------------------- Rval is in ascending order */
    /*-------------------- Rval[0] is smallest, Rval[k-1] is largest */
    /*-------------------- special vector for TR that is the bottom row of 
                           eigenvectors of Tm */
    s[0] = beta * EvecT[k-1];
    s[1] = beta * EvecT[(k-1)*lanm1_l+(k-1)];
    /*---------------------- bounds */
    if (bndtype <= 1) {
      /*-------------------- BOUNDS type 1 (simple) */
      t1 = fabs(s[0]);
      t2 = fabs(s[1]);
    } else {
      /*-------------------- BOUNDS type 2 (Kato-Temple) */
      t1 = 2.0*s[0]*s[0] / (Rval[1] - Rval[0]);
      t2 = 2.0*s[1]*s[1] / (Rval[k-1] - Rval[k-2]);
    }
    lmin = Rval[0]   - t1;
    lmax = Rval[k-1] + t2;
    /*---------------------- Compute two Ritz vectors: 
     *                       Rvec(:,1) = V(:,1:k) * EvecT(:,1) 
     *                       Rvec(:,end) = V(:,1:k) * EvecT(:,end) */
    pEVSL_ParvecsGemv(1.0, V, k, EvecT, 0.0, Rvec);
    pEVSL_ParvecsGemv(1.0, V, k, EvecT+(k-1)*lanm1_l, 0.0, Rvec+1);
    if (ifGenEv) {
      pEVSL_ParvecsGemv(1.0, Z, k, EvecT, 0.0, BRvec);
      pEVSL_ParvecsGemv(1.0, Z, k, EvecT+(k-1)*lanm1_l, 0.0, BRvec+1);
    }
    /*---------------------- Copy two Rval and Rvec to TR set */
    trlen = 2;
    for (i=0; i<2; i++) {
      pevsl_Parvec *y = Rvec + i;
      pEVSL_ParvecsGetParvecShell(V, i, v);
      pEVSL_ParvecCopy(y, v);
      if (ifGenEv) {
        pevsl_Parvec *By = BRvec + i;
        pEVSL_ParvecsGetParvecShell(Z, i, z);
        pEVSL_ParvecCopy(By, z);
      }
    }
    Rval[1] = Rval[k-1];
    /*-------------------- recompute residual norm for debug only */
#if COMP_RES
#else
    if (do_print) {
      pEVSL_fprintf0(rank, fstats,"it %4d, k %3d: ritz %.15e %.15e, t1,t2 %e %e, res %.15e %.15e\n", 
                    it, k, Rval[0], Rval[1], t1, t2, fabs(s[0]), fabs(s[1]));
    }
#endif
    /*---------------------- test convergence */
    if (t1+t2 < tol*(fabs(lmin)+fabs(lmax))) {
      break;
    }
    /*-------------------- prepare to restart.  First zero out all T */
    memset(T, 0, lanm1_l*lanm1_l*sizeof(double));
    /*-------------------- move starting vector vector V(:,k+1);  V(:,trlen+1) = V(:,k+1) */
    pEVSL_ParvecsGetParvecShell(V, k, v);
    pEVSL_ParvecsGetParvecShell(V, trlen, vnew);
    pEVSL_ParvecCopy(v, vnew);
    if (ifGenEv) {
      pEVSL_ParvecsGetParvecShell(Z, k, z);
      pEVSL_ParvecsGetParvecShell(Z, trlen, znew);
      pEVSL_ParvecCopy(z, znew);
    }
  } /* outer loop (it) */

  /*-------------------- Done.  output : */
  *lammin = lmin;
  *lammax = lmax;
  /*-------------------- free arrays */
  pEVSL_ParvecsFree(V);
  PEVSL_FREE(V);
  PEVSL_FREE(T);
  PEVSL_FREE(Rval);
  PEVSL_FREE(EvecT);
  pEVSL_ParvecFree(&Rvec[0]);
  pEVSL_ParvecFree(&Rvec[1]);
  PEVSL_FREE(warr);
  if (ifGenEv) {
    pEVSL_ParvecsFree(Z);
    PEVSL_FREE(Z);
    pEVSL_ParvecFree(&BRvec[0]);
    pEVSL_ParvecFree(&BRvec[1]);
  }

  double tme = pEVSL_Wtime();
  pevsl->stats->t_eigbounds += tme - tms;
  
  return 0;
}
