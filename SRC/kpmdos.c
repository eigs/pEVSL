#include "pevsl_int.h"

/**
 * @file kpmdos.c
 * @brief Compute DOS by KPM methods
 */

/**----------------------------------------------------------------------
 *
 * @brief This function  computes the  coefficients of the  density of
 * states  in  the  chebyshev   basis.   It  also  returns  the
 * estimated number of eigenvalues in the interval given by intv.
 * @param Mdeg     degree of polynomial to be used. 
 * @param damping  type of damping to be used [0=none,1=jackson,2=sigma]
 * @param nvec     number of random vectors to use for sampling
 * @param intv   an array of length 4  \n
 *                 [intv[0] intv[1]] is the interval of desired eigenvalues 
 *                 that must be cut (sliced) into n_int  sub-intervals \n
 *                 [intv[2],intv[3]] is the global interval of eigenvalues 
 *                 it must contain all eigenvalues of A \n
 * @param[int] ngroups : number of groups
 * @param[int] groupid : rank of this group
 * @param[int] gl_comm : group-leader communicator
 *             pEVSL_Kpmdos can be ran with multiple groups of procs, where in
 *             each group, there is a pEVSL instance. In this case, provide a
 *             MPI_Comm of the group-leaders (with group rank 0). Results will
 *             computed collectively.
 *             Otherwise, set 
 *             ngroups = 1, groupid = 0, and gl_comm := MPI_COMM_NULL
 * @param[out] mu   array of Chebyshev coefficients [of size Mdeg+1]
 * @param[out] ecnt estimated num of eigenvalues in the interval of interest
 *
 *----------------------------------------------------------------------*/

int pEVSL_Kpmdos(pevsl_Data *pevsl, int Mdeg, int damping, int nvec, double *intv,
                 int ngroups, int groupid, MPI_Comm gl_comm, double *mu, double *ecnt) {
  
  const int ifGenEv = pevsl->ifGenEv;
  /*-------------------- MPI comm of this instance of pEVSL */
  MPI_Comm comm = pevsl->comm;
  int N = pevsl->N;
  int n = pevsl->n;
  int nfirst = pevsl->nfirst;

  int rank;
  MPI_Comm_rank(comm, &rank);

  /*-------------------- parvec needed */
  pevsl_Parvec parvec[5], *v, *vkm1, *vk, *vkp1, *w = NULL, *tmp;
  pEVSL_ParvecCreate(N, n, nfirst, comm, &parvec[0]);
  pEVSL_ParvecCreate(N, n, nfirst, comm, &parvec[1]);
  pEVSL_ParvecCreate(N, n, nfirst, comm, &parvec[2]);
  pEVSL_ParvecCreate(N, n, nfirst, comm, &parvec[3]);
  v    = parvec;
  vkm1 = parvec + 1;
  vk   = parvec + 2;
  vkp1 = parvec + 3;
  /*-------------------- workspace for generalized eigenvalue prob */
  if (ifGenEv) {
    pEVSL_ParvecCreate(N, n, nfirst, comm, &parvec[4]);
    w = parvec + 4;
  }
  
  double *jac, ctr, wid, scal, t, tcnt, beta1, beta2, aa, bb;
  int k, k1, m, mdegp1, one=1, vec_start, vec_end;

  PEVSL_MALLOC(jac, Mdeg+1, double);
  /*-------------------- check if the interval is valid */
  if (check_intv(intv, stdout) < 0) {
    return -1;
  }
  aa = PEVSL_MAX(intv[0], intv[2]);  bb = PEVSL_MIN(intv[1], intv[3]);
  if (intv[0] < intv[2] || intv[1] > intv[3]) {
    fprintf(stdout, " warning [%s (%d)]: interval (%e, %e) is adjusted to (%e, %e)\n",
            __FILE__, __LINE__, intv[0], intv[1], aa, bb);
  }

  /*-------------------- some needed constants */
  ctr  = (intv[3]+intv[2])/2.0;
  wid  = (intv[3]-intv[2])/2.0;
  t = PEVSL_MAX(-1.0 + DBL_EPSILON, (aa-ctr)/wid);
  beta1 = acos(t);
  t = PEVSL_MIN( 1.0 - DBL_EPSILON, (bb-ctr)/wid);
  beta2 = acos(t);
  /*-------------------- compute damping coefs. */
  dampcf(Mdeg, damping, jac);
  /*-------------------- readjust jac[0] it was divided by 2 */
  jac[0] = 1.0;
  memset(mu, 0, (Mdeg+1)*sizeof(double));
  /*-------------------- tcnt: total count */
  tcnt = 0.0;
  /*-------------------- if we have more than one groups, 
   *                     partition nvecs among groups */
  if (ngroups > 1) {
    pEVSL_Part1d(nvec, ngroups, &groupid, &vec_start, &vec_end, 1);
  } else {
    vec_start = 0;
    vec_end = nvec;
  }
  /*-------------------- random vectors loop */
  for (m = vec_start; m < vec_end; m++) {
    if (ifGenEv) {
      /* unit 2-norm v */
      pEVSL_ParvecRand(v);
      pEVSL_ParvecNrm2(v, &t);
      pEVSL_ParvecScal(v, 1.0 / t);  
      /*  w = L^{-T}*v */
      pEVSL_SolveLT(pevsl, v, w);
      /* v = B*w */
      pEVSL_MatvecB(pevsl, w, v);
      pEVSL_ParvecDot(v, w, &t);
      pEVSL_ParvecCopy(w, vk);
    } else {
      /* unit 2-norm */
      pEVSL_ParvecRand(v);
      pEVSL_ParvecNrm2(v, &t);
      pEVSL_ParvecScal(v, 1.0 / t);  
      pEVSL_ParvecCopy(v, vk);
    }
    
    pEVSL_ParvecSetZero(vkm1);
    mu[0] += jac[0];
    //-------------------- for eigCount
    tcnt -= jac[0]*(beta2-beta1);  
    /*-------------------- Chebyshev (degree) loop */
    for (k=0; k<Mdeg; k++) {
      /*-------------------- Cheb. recurrence */
      if (ifGenEv) {
        /* v_{k+1} := B \ A * v_k (partial result) */
        pEVSL_MatvecA(pevsl, vk, w);
        pEVSL_SolveB(pevsl, w, vkp1);
      } else {
        pEVSL_MatvecA(pevsl, vk, vkp1);
      }
      scal = k == 0 ? 1.0 : 2.0;
      scal /= wid;

      pEVSL_ParvecAxpy(-ctr, vk, vkp1);
      pEVSL_ParvecScal(vkp1, scal);
      pEVSL_ParvecAxpy(-1.0, vkm1, vkp1);
      /*-------------------- rotate pointers to exchange vectors */
      tmp = vkm1;
      vkm1 = vk;
      vk = vkp1;
      vkp1 = tmp;
      /*-------------------- accumulate dot products for DOS expansion */
      k1 = k+1;
      pEVSL_ParvecDot(vk, v, &t);
      t *= 2.0 * jac[k1];
      mu[k1] += t;
      /*-------------------- for eig. counts */
      tcnt -= t*(sin(k1*beta2)-sin(k1*beta1))/k1;  
    }
  } /* the end of random vectors loop */

  /* if we have more than one group */
  if (ngroups > 1) {
    double *mu_global, tcnt_global;
    PEVSL_MALLOC(mu_global, Mdeg+1, double);
    /* Sum of all partial results: the group leaders first do an All-reduce. 
       Then, group leaders will do broadcasts
     */
    if (rank == 0) {
      MPI_Allreduce(mu, mu_global, Mdeg+1, MPI_DOUBLE, MPI_SUM, gl_comm);
      MPI_Allreduce(&tcnt, &tcnt_global, 1, MPI_DOUBLE, MPI_SUM, gl_comm);
    }
    MPI_Bcast(mu_global, Mdeg+1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&tcnt_global, 1, MPI_DOUBLE, 0, comm);

    tcnt = tcnt_global;
    memcpy(mu, mu_global, (Mdeg+1)*sizeof(double));
    PEVSL_FREE(mu_global);
  }

  /*-------------------- change of interval + scaling in formula */
  t = 1.0 /(((double)nvec)*PI);
  mdegp1 = Mdeg+1;
  DSCAL(&mdegp1, &t, mu, &one);
  tcnt *= t * ((double) N);
  *ecnt = tcnt;
  /*-------------------- dealloc memory */
  pEVSL_ParvecFree(&parvec[0]);
  pEVSL_ParvecFree(&parvec[1]);
  pEVSL_ParvecFree(&parvec[2]);
  pEVSL_ParvecFree(&parvec[3]);
  if (ifGenEv) {
    pEVSL_ParvecFree(&parvec[4]);
  }
  PEVSL_FREE(jac);

  return 0;
}

