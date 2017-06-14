#include "pevsl_protos.h"

#define COMP_RES 0
/**
 * @brief Lanczos process for eigenvalue bounds [Thick restart version]
 *
 * @param lanm      Dimension of Krylov subspace [restart dimension]
 * 
 * @param maxit  max Num of outer Lanczos iterations (restarts) allowed -- 
 *         Each restart may or use the full lanm lanczos steps or fewer.
 * 
 * @param tol       tolerance for convergence
 * @param vinit     initial  vector for Lanczos -- [optional]
 *
 * @b Modifies:
 * @param[out] lammin   Lower bound of the spectrum
 * @param[out] lammax   Upper bound of the spectrum
 * @param[out] fstats File stream which stats are printed to
 *
 * @return Returns 0 on success 
 *
 **/
int pEVSL_LanTrbounds(int lanm, int maxit, double tol, pevsl_Parvec *vinit,
                      int bndtype, double *lammin, double *lammax, MPI_Comm comm, FILE *fstats) {
  double tms = pEVSL_Wtime();
  double lmin=0.0, lmax=0.0, t, t1, t2;
  int do_print = 1, rank;
  /* handle case where fstats is NULL. Then no output. Needed for openMP. */
  if (fstats == NULL) {
    do_print = 0;
  }
  /*-------------------- MPI rank in comm */
  MPI_Comm_rank(comm, &rank);  
  /* size of the matrix */
  int N;
  N = pevsl_data.N;
  /*--------------------- adjust lanm and maxit */
  lanm = PEVSL_MIN(lanm, N);
  int lanm1=lanm+1;
  /*  if use full lanczos, should not do more than n iterations */
  if (lanm == N) {
    maxit = PEVSL_MIN(maxit, N);
  }
  /*--------------------   some constants frequently used */
  /* char cT='T'; */
  //char cN = 'N';
  //int one = 1;
  //double done=1.0, dmone=-1.0, dzero=0.0;
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
  pevsl_Parvec *V, *Z;
  double *T;
  PEVSL_MALLOC(V, lanm1, pevsl_Parvec);
  for (i=0; i<lanm1; i++) {
    pEVSL_ParvecDupl(vinit, &V[i]);
  }
  /*-------------------- for gen eig prob, storage for Z = B * V */
  if (pevsl_data.ifGenEv) {
    PEVSL_MALLOC(Z, lanm1, pevsl_Parvec);
    for (i=0; i<lanm1; i++) {
      pEVSL_ParvecDupl(vinit, &Z[i]);
    }
  } else {
    Z = V;
  }
  /*-------------------- T must be zeroed out initially */
  PEVSL_CALLOC(T, lanm1*lanm1, double);
  /*-------------------- trlen = dim. of thick restart set */
  int trlen = 0;
  /*-------------------- Ritz values and vectors of p(A) */
  double *Rval;
  PEVSL_MALLOC(Rval, lanm, double);
  /*-------------------- Only compute 2 Ritz vectors */
  pevsl_Parvec Rvec[2], BRvec[2];
  pEVSL_ParvecDupl(vinit, &Rvec[0]);
  pEVSL_ParvecDupl(vinit, &Rvec[1]);
  if (pevsl_data.ifGenEv) {
    pEVSL_ParvecDupl(vinit, &BRvec[0]);
    pEVSL_ParvecDupl(vinit, &BRvec[1]);
  }
  /*-------------------- Eigen vectors of T */
  double *EvecT;
  PEVSL_MALLOC(EvecT, lanm1*lanm1, double);
  /*-------------------- s used by TR (the ``spike'' of 1st block in Tm)*/
  double s[3];
  /*-------------------- alloc some work space */
  //double *work;
  //int work_size = 2*n;
  //Malloc(work, work_size, double);  
  /*-------------------- copy initial vector to V(:,1)   */
  pEVSL_ParvecCopy(vinit, V);
  /*-------------------- normalize it */
  if (pevsl_data.ifGenEv) {
    /* B norm */
    pEVSL_MatvecB(V, Z);
    pEVSL_ParvecDot(V, Z, &t);
    t = 1.0 / sqrt(t);
    /* z = B*v */
    pEVSL_ParvecScal(Z, t);    
  } else {
    /* 2-norm */
    pEVSL_ParvecNrm2(V, &t);
    t = 1.0 / t;
  }
  /* unit B-norm or 2-norm */
  pEVSL_ParvecScal(V, t);
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
      pevsl_Parvec *v = V + k1;
      pevsl_Parvec *z = Z + k1;
      /*------------------ next Lanczos vector */
      pevsl_Parvec *vnew = v + 1;
      pevsl_Parvec *znew = z + 1;
      /*------------------ znew = A * v */
      pEVSL_MatvecA(v, znew);
      /*-------------------- restart with 'trlen' Ritz values/vectors
                             T = diag(Rval(1:trlen)) */
      for (i=0; i<trlen; i++) {
        T[i*lanm1+i] = Rval[i];
        wn += fabs(Rval[i]);
      }
      /*--------------------- s(k) = V(:,k)'* znew */
      pEVSL_ParvecDot(v, znew, s+k1);
      /*--------------------- znew = znew - Z(:,1:k)*s(1:k) */
      for (i=0; i<k; i++) {
        pEVSL_ParvecAxpy(-s[i], Z+i, znew);
      }
      /*-------------------- expand T matrix to k-by-k, arrow-head shape
                             T = [T, s(1:k-1)] then T = [T; s(1:k)'] */
      for (i=0; i<k1; i++) {
        T[trlen*lanm1+i] = s[i];
        T[i*lanm1+trlen] = s[i];
        wn += 2.0 * fabs(s[i]);
      }
      T[trlen*lanm1+trlen] = s[k1];
      wn += fabs(s[k1]);
      if (pevsl_data.ifGenEv) {
        /*-------------------- vnew = B \ znew */
        pEVSL_SolveB(znew, vnew);
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
        if (pevsl_data.ifGenEv) {
          /* vnew = vnew - V(:,1:k)*Z(:,1:k)'*vnew */
          MGS_DGKS2(k, NGS_MAX, V, Z, vnew);          
          pEVSL_MatvecB(vnew, znew);
          pEVSL_ParvecDot(vnew, znew, &beta);
          beta = sqrt(beta);         
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          pEVSL_ParvecScal(znew, ibeta);
          beta = 0.0;            
        } else {
          /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
          /*   beta = norm(w) */
          MGS_DGKS(k, NGS_MAX, V, vnew, &beta);
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          beta = 0.0;
        }
      } else {
        /*------------------- w = w / beta */
        double ibeta = 1.0 / beta;
        pEVSL_ParvecScal(vnew, ibeta);
        if (pevsl_data.ifGenEv) {
          pEVSL_ParvecScal(znew, ibeta);
        }
      }
      /*------------------- T(k+1,k) = beta; T(k,k+1) = beta; */
      T[k1*lanm1+k] = beta;
      T[k*lanm1+k1] = beta;
    } /* if (trlen > 0) */
    /*-------------------- Done with TR step. Rest of Lanczos step */
    /*-------------------- regardless of trlen, *(k+1)* is the current 
     *                     number of Lanczos vectors in V */
    /*-------------------- pointer to the previous Lanczos vector */
    pevsl_Parvec *zold = k > 0 ? Z+k-1 : NULL;
    /*------------------------------------------------------*/
    /*------------------ Lanczos inner loop ----------------*/
    /*------------------------------------------------------*/
    while (k < lanm && it < maxit) {
      k++;
      /*---------------- a quick reference to V(:,k) */
      pevsl_Parvec *v = &V[k-1];
      pevsl_Parvec *z = &Z[k-1];
      /*---------------- next Lanczos vector */
      pevsl_Parvec *vnew = v + 1;
      pevsl_Parvec *znew = z + 1;
      /*------------------ znew = A * v */
      pEVSL_MatvecA(v, znew);
      it++;
      /*-------------------- znew = znew - beta*zold */
      if (zold) {
        pEVSL_ParvecAxpy(-beta, zold, znew);
      }
      /*-------------------- alpha = znew'*v */
      double alpha;
      pEVSL_ParvecDot(v, znew, &alpha);
      /*-------------------- T(k,k) = alpha */
      T[(k-1)*lanm1+(k-1)] = alpha;
      wn += fabs(alpha);
      /*-------------------- znew = znew - alpha*z */
      pEVSL_ParvecAxpy(-alpha, z, znew);
      /*-------------------- FULL reortho to all previous Lan vectors */
      if (pevsl_data.ifGenEv) {
        /* znew = znew - Z(:,1:k)*V(:,1:k)'*znew */
        MGS_DGKS2(k, NGS_MAX, Z, V, znew);
        /* vnew = B \ znew */
        pEVSL_SolveB(znew, vnew);
        /*-------------------- beta = (vnew, znew)^{1/2} */
        pEVSL_ParvecDot(vnew, znew, &beta);
        beta = sqrt(beta);
      } else {
        /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
        /*   beta = norm(w) */
        MGS_DGKS(k, NGS_MAX, V, vnew, &beta);
      }
      wn += 2.0 * beta;
      nwn += 3;
      /*-------------------- zold = z */
      zold = z;
      /*-------------------- lucky breakdown test */
      if (beta*nwn < orthTol*wn) {
        if (do_print) {
          pEVSL_fprintf0(rank, fstats, "it %4d: Lucky breakdown, beta = %.15e\n", it, beta);
        }
        /* generate a new init vector */
        pEVSL_ParvecRand(vnew);
        if (pevsl_data.ifGenEv) {
          /* vnew = vnew - V(:,1:k)*Z(:,1:k)'*vnew */
          MGS_DGKS2(k, NGS_MAX, V, Z, vnew);          
          pEVSL_MatvecB(vnew, znew);
          pEVSL_ParvecDot(vnew, znew, &beta);
          beta = sqrt(beta); 
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          pEVSL_ParvecScal(znew, ibeta);
          beta = 0.0;            
        } else {
          /*   vnew = vnew - V(:,1:k)*V(:,1:k)'*vnew */
          /*   beta = norm(w) */
          MGS_DGKS(k, NGS_MAX, V, vnew, &beta);          
          double ibeta = 1.0 / beta;
          pEVSL_ParvecScal(vnew, ibeta);
          beta = 0.0;
        }
      } else {
        /*---------------------- vnew = vnew / beta */
        double ibeta = 1.0 / beta;
        pEVSL_ParvecScal(vnew, ibeta);
        if (pevsl_data.ifGenEv) {
          /*-------------------- znew = znew / beta */
          pEVSL_ParvecScal(znew, ibeta);
        }
      }
      /*-------------------- T(k,k+1) = T(k+1,k) = beta */
      T[k*lanm1+(k-1)] = beta;
      T[(k-1)*lanm1+k] = beta;
    } /* while (k<mlan) loop */

    /*-------------------- solve eigen-problem for T(1:k,1:k)
                           vals in Rval, vecs in EvecT */
    SymEigenSolver(k, T, lanm1, EvecT, lanm1, Rval);

    /*-------------------- Rval is in ascending order */
    /*-------------------- Rval[0] is smallest, Rval[k-1] is largest */
    /*-------------------- special vector for TR that is the bottom row of 
                           eigenvectors of Tm */
    s[0] = beta * EvecT[k-1];
    s[1] = beta * EvecT[(k-1)*lanm1+(k-1)];
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
    pEVSL_ParvecSetZero(&Rvec[0]);
    for (i=0; i<k; i++) {
      pEVSL_ParvecAxpy(EvecT[i], V+i, &Rvec[0]);
    }
    pEVSL_ParvecSetZero(&Rvec[1]);     
    for (i=0; i<k; i++) {
      pEVSL_ParvecAxpy(EvecT[(k-1)*lanm1+i], V+i, &Rvec[1]);
    }      
    if (pevsl_data.ifGenEv) {
      pEVSL_ParvecSetZero(&BRvec[0]);
      for (i=0; i<k; i++) {
        pEVSL_ParvecAxpy(EvecT[i], Z+i, &BRvec[0]);
      }
      pEVSL_ParvecSetZero(&BRvec[1]);     
      for (i=0; i<k; i++) {
        pEVSL_ParvecAxpy(EvecT[(k-1)*lanm1+i], Z+i, &BRvec[1]);
      }      
    }
    /*---------------------- Copy two Rval and Rvec to TR set */
    trlen = 2;
    for (i=0; i<2; i++) {
      pevsl_Parvec *y = Rvec + i;
      pEVSL_ParvecCopy(y, V+i);
      if (pevsl_data.ifGenEv) {
        pevsl_Parvec *By = BRvec + i;
        pEVSL_ParvecCopy(By, Z+i);
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
    memset(T, 0, lanm1*lanm1*sizeof(double));
    /*-------------------- move starting vector vector V(:,k+1);  V(:,trlen+1) = V(:,k+1) */
    pEVSL_ParvecCopy(V+k, V+trlen);
    if (pevsl_data.ifGenEv) {
      pEVSL_ParvecCopy(Z+k, Z+trlen);
    }
  } /* outer loop (it) */

  /*-------------------- Done.  output : */
  *lammin = lmin;
  *lammax = lmax;
  /*-------------------- free arrays */
  for (i=0; i<lanm1; i++) {
    pEVSL_ParvecFree(&V[i]);
  }
  PEVSL_FREE(V);
  PEVSL_FREE(T);
  PEVSL_FREE(Rval);
  PEVSL_FREE(EvecT);
  pEVSL_ParvecFree(&Rvec[0]);
  pEVSL_ParvecFree(&Rvec[1]);
  //free(work);
  if (pevsl_data.ifGenEv) {
    for (i=0; i<lanm1; i++) {
      pEVSL_ParvecFree(&Z[i]);
    }
    PEVSL_FREE(Z);
    pEVSL_ParvecFree(&BRvec[0]);
    pEVSL_ParvecFree(&BRvec[1]);
  }

  double tme = pEVSL_Wtime();
  pevsl_stat.t_eigbounds += tme - tms;
  
  return 0;
}

