#include "pevsl_int.h"

/**  
 * @brief Computes the integrals \f$\int_{xi[0]}^{xi[j]} p(t) dt\f$
 *  where p(t) is the approximate DOS as given in the KPM method
 *  in the expanded form:  \f$\sum mu_i C_i /\sqrt{1-t^2}\f$
 **/
void intChx(const int Mdeg, double *mu, const int npts, double *xi, double *yi) {
  //
  int ndp1, j, k;
  if(npts <= 0) {
    fprintf(stderr, "Must have more than 0 points");
    exit (1);
  }
  double val0, theta0, *thetas;
  PEVSL_MALLOC(thetas, npts, double);
  ndp1   = Mdeg+1; 
  //  if (xi[0]<-1.0) xi[0] = -1; 
  //if (xi[npts-1]> 1.0) xi[npts-1]  = 1; 

  for (j=0; j<npts; j++)
    thetas[j] = acos(xi[j]);
  theta0 = thetas[0];
  for (j=0; j<npts; j++) 
    yi[j] = mu[0]*(theta0 - thetas[j]);
  //-------------------- degree loop  
  for (k=1; k<ndp1; k++){
    val0 = sin(k*theta0)/k;
    //-------------------- points loop
    for (j=0; j<npts; j++)
      yi[j] += mu[k]*(val0 - sin(k*thetas[j])/k);
  }
  PEVSL_FREE (thetas);
}

/**----------------------------------------------------------------------- 
 * @brief given the dos function defined by mu find a partitioning
 * of sub-interval [a,b] of the spectrum so each 
 * subinterval has about the same number of eigenvalues
 * Mdeg = degree.. mu is of length Mdeg+1  [0---> Mdeg]
 * on return [ sli[i],sli[i+1] ] is a subinterval (slice).
 *
 * @param *sli  see above (output)
 * @param *mu   coeffs of polynomial (input)
 * @param Mdeg     degree of polynomial (input)
 * @param *intv  an array of length 4 
 *                [intv[0] intv[1]] is the interval of desired eigenvalues
 *                that must be cut (sliced) into n_int  sub-intervals
 *                [intv[2],intv[3]] is the global interval of eigenvalues
 *                it must contain all eigenvalues of A
 * @param n_int   number of slices wanted (input)
 * @param npts      number of points to use for discretizing the interval
 *                [a b]. The more points the more accurate the intervals. 
 *                it is recommended to set npts to a few times the number 
 *                of eigenvalues in the interval [a b] (input). 
 *
 *----------------------------------------------------------------------*/
int pEVSL_Spslicer(double *sli, double *mu, int Mdeg, double *intv, int n_int, int npts) {
  int ls, ii, err=0;
  double  ctr, wid, aL, bL, target, aa, bb;

  if (check_intv(intv, stdout) < 0) {
    return -1;
  }

  // adjust a, b: intv[0], intv[1]
  aa = PEVSL_MAX(intv[0], intv[2]);  bb = PEVSL_MIN(intv[1], intv[3]);
  if (intv[0] < intv[2] || intv[1] > intv[3]) {
    fprintf(stdout, " warning [%s (%d)]:  interval (%e, %e) is adjusted to (%e, %e)\n",
        __FILE__, __LINE__, intv[0], intv[1], aa, bb);
  }

  //-------------------- 
  memset(sli,0,(n_int+1)*sizeof(double));
  //-------------------- transform to ref interval [-1 1]
  //-------------------- take care of trivial case n_int==1
  if (n_int == 1){
    sli[0] = intv[0];
    sli[1] = intv[1];
    return 0;
  }
  //-------------------- general case 
  ctr = (intv[3] + intv[2])/2;
  wid = (intv[3] - intv[2])/2;
  aL  = (aa - ctr)/wid;   // (a - ctr)/wid 
  bL  = (bb - ctr)/wid;   // (b - ctr)/wid
  aL = PEVSL_MAX(aL, -1.0);
  bL = PEVSL_MIN(bL,  1.0);
  npts = PEVSL_MAX(npts, 2*n_int+1);
  double *xi, *yi;
  PEVSL_MALLOC(xi, npts, double);
  PEVSL_MALLOC(yi, npts, double);
  linspace(aL, bL, npts, xi);
  //printf(" aL %15.3e bL %15.3e \n",aL,bL);
  //-------------------- get all integrals at the xi's 
  //-------------------- exact integrals used.
  intChx(Mdeg, mu, npts, xi, yi) ; 
  //-------------------- goal: equal share of integral per slice
  target = yi[npts-1] / (double)n_int;
  ls = 0;
  ii = 0;
  // use the unadjust left boundary
  sli[ls] = intv[0];
  //-------------------- main loop 
  while (++ls < n_int) {
    while (ii < npts && yi[ii] < target) {
      ii++;
    }
    if (ii == npts) {
      break;
    }
    //-------------------- take best of 2 points in interval
    if ( (target-yi[ii-1]) < (yi[ii]-target)) {
      ii--;
    }
    sli[ls] = ctr + wid*xi[ii];
    //-------------------- update target.. Slice size adjusted
    target = yi[ii] + (yi[npts-1] - yi[ii])/(n_int-ls);
    //printf("ls %d, n_int %d, target %e\n", ls, n_int, target);
  }

  // use the unadjust left boundary
  sli[n_int] = intv[1];

  //-------------------- check errors
  if (ls != n_int) {
    err = 1;
  }
  for (ii=1; ii<=n_int; ii++) {
    if (sli[ii] < sli[ii-1]) {
      err += 2;
      break;
    }
  }
  
  /*
  if (err) {
    printf("sli:\n");
    for (ii=0; ii<=n_int; ii++) {
      printf("%.15f\n", sli[ii]);
    }
    printf("\n");
    save_vec(npts, xi, "OUT/xi.out");
    save_vec(npts, yi, "OUT/yi.out");
  }
  */

  /*-------------------- free arrays */
  PEVSL_FREE(xi);
  PEVSL_FREE(yi);
  return err;
}
