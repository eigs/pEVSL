#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define OUTPUT_LEN 32768

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and computes all eigenvalues in a given interval [a  b]
  The default set values are
  nx = 41; ny = 53; nz = 1;
  a = 0.4; b = 0.8;
  nslices = 1 [one slice only] 
  other parameters 
  tol [tolerance for stopping - based on residual]
  Mdeg = pol. degree used for DOS
  nvec  = number of sample vectors used for DOS
  This uses:
  Non-restart Lanczos with polynomial filtering
------------------------------------------------------------*/
  int n, nx, ny, nz, i, j, npts, nslices, nvec, Mdeg, nev, 
      ngroups, mlan, ev_int, sl, flg, ierr, np, rank;
  /* find the eigenvalues of A in the interval [a,b] */
  double a, b, lmax, lmin, ecount, tol, *sli, *mu;
  double xintv[4];
  double tm;
  char *msg = NULL;
  /*-------------------- communicator struct, which contains all the communicators */
  CommInfo comm;
  pevsl_Parvec vinit;
  pevsl_Polparams pol;
  FILE *fstats = NULL;
  /*--------------------- Initialize MPI */
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  /*-------------------- size and rank */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*-------------------- matrix A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- instance of pEVSL */
  pevsl_Data *pevsl;
  /*-------------------- default values */
  nx = 16;
  ny = 16;
  nz = 20;
  a  = 0.4;
  b  = 0.8;
  nslices = 4;
  ngroups = 3;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -nslices [int] -ngroups [int] -a [double] -b [double]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("a", DOUBLE, &a, argc, argv);
  flg = findarg("b", DOUBLE, &b, argc, argv);
  flg = findarg("nslices", INT, &nslices, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  /*-------------------- eigenvalue bounds set by hand */
  lmin = 0.0;  
  lmax = nz == 1 ? 8.0 : 12.0;
  xintv[0] = a;
  xintv[1] = b;
  xintv[2] = lmin;
  xintv[3] = lmax;
  tol  = 1e-8;
  n = nx * ny * nz;
  /*-------------------- Partition MPI_COMM_WORLD into ngroups subcomm's,
   * create a communicator for each group, and one for group leaders
   * saved in comm */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  /*-------------------- Group leader (group_rank == 0) creates output file */
  if (comm.group_rank == 0) {
    /* output on screen */
    PEVSL_CALLOC(msg, OUTPUT_LEN, char);
    char fname[1024];
    sprintf(fname, "OUT/LapPLanN_G%d.out", comm.group_id);
    if (!(fstats = fopen(fname,"w"))) {
      PEVSL_ABORT(MPI_COMM_WORLD, -2, " failed in opening output file in OUT/\n");
    }
    fprintf(fstats, " nx %d ny %d nz %d, nslices %d, a = %e b = %e\n", 
            nx, ny, nz, nslices, a, b);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrix A
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  /*-------------------- start pEVSL 
   * Create an instance of pEVSL on the GROUP communicator */
  pEVSL_Start(comm.comm_group, &pevsl);
  /*-------------------- set matrix A */
  pEVSL_SetAParcsr(pevsl, &A);
  /*-------------------- call DOS */
  Mdeg = 300;
  nvec = 60;
  mu = (double *) malloc((Mdeg+1)*sizeof(double));
#if 0
  tm = pEVSL_Wtime();
  ierr = pEVSL_Kpmdos(pevsl, Mdeg, 1, nvec, xintv, comm.ngroups, comm.group_id,
                      comm.comm_group_leader, mu, &ecount);
  tm = pEVSL_Wtime() - tm;
  if (ierr) {
    printf("kpmdos error %d\n", ierr);
    return 1;
  }
  if (comm.group_rank == 0) {
    fprintf(fstats, " Time to build DOS (kpmdos) was : %10.2f  \n", tm);
    fprintf(fstats, " estimated eig count in interval: %.15e \n", ecount);
  }
  /*-------------------- call Spslicer to slice the spectrum */
  npts = 10 * ecount;
  sli = (double *) malloc((nslices+1)*sizeof(double));
  if (comm.group_rank == 0) {
    fprintf(fstats, " DOS parameters: Mdeg = %d, nvec = %d, npnts = %d\n", Mdeg, nvec, npts);
  }
  ierr = pEVSL_SpslicerKpm(sli, mu, Mdeg, xintv, nslices,  npts);
  /*-------------------- slicing done */
  if (ierr) {
    printf("spslicer error %d\n", ierr);
    return 1;
  }
#else
  int msteps = 40;
  npts = 200;
  double *xdos = (double *)calloc(npts, sizeof(double));
  double *ydos = (double *)calloc(npts, sizeof(double));
  tm = pEVSL_Wtime();
  pEVSL_LanDosG(pevsl, nvec, msteps, npts, xdos, ydos, &ecount, xintv,
                comm.ngroups, comm.group_id, comm.comm_group_leader);
  tm = pEVSL_Wtime() - tm;
  fprintf(stdout, " estimated eig count in interval: %f \n", ecount);
  if (comm.group_rank == 0) {
    fprintf(fstats, " Time to build DOS (Lanczos dos) was : %10.2f  \n", tm);
    fprintf(fstats, " estimated eig count in interval: %.15e \n", ecount);
  }
  /*-------------------- call Spslicer to slice the spectrum */
  sli = (double *) malloc((nslices+1)*sizeof(double));
  pEVSL_SpslicerLan(xdos, ydos, nslices, npts, sli);
  PEVSL_FREE(xdos);
  PEVSL_FREE(ydos);
#endif

  if (comm.group_rank == 0) {
    fprintf(fstats, "====================  SLICES FOUND  ====================\n");
    for (j=0; j<nslices;j++) {
      fprintf(fstats, " %2d: [% .15e , % .15e]\n", j+1, sli[j],sli[j+1]);
    }
    fprintf(fstats, "========================================================\n");
  }
  /*--------------------- print stats */
  pEVSL_StatsPrint(pevsl, fstats);
  /*-------------------- # eigs per slice */
  ev_int = (int) (1 + ecount / ((double) nslices));
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
  pEVSL_ParvecRand(&vinit);

  /*------------------- For each slice call ChebLanr, each group pick one slice */
  for (sl=comm.group_id; sl<nslices; sl+=comm.ngroups) {
    int nev2, *ind, nev_ex;
    double *lam, *res, *lam_ex, ai, bi;
    pevsl_Parvecs *Y;
    /*-------------------- */
    pEVSL_StatsReset(pevsl);
    ai = sli[sl];
    bi = sli[sl+1];
    /*-------------------- approximate number of eigenvalues wanted */
    nev = ev_int+2;
    /*-------------------- Dimension of Krylov subspace */
    mlan = max(5*nev, 300);
    mlan = min(mlan, n);
    /*-------------------- Interval */
    xintv[0] = ai;    xintv[1] = bi;  
    xintv[2] = lmin;  xintv[3] = lmax;
    //-------------------- set up default parameters for pol.      
    pEVSL_SetPolDef(&pol);
    //-------------------- this is to show how you can reset some of the
    //                     parameters to determine the filter polynomial
    pol.damping = 2;
    //-------------------- use a stricter requirement for polynomial
    pol.thresh_int = 0.8;
    pol.thresh_ext = 0.2;
    pol.max_deg  = 3000;
    // pol.deg = 20 //<< this will force this exact degree . not recommended
    //                   it is better to change the values of the thresholds
    //                   pol.thresh_ext and plot.thresh_int
    //-------------------- Now determine polymomial to use
    pEVSL_FindPol(xintv, &pol);

    if (comm.group_rank == 0) {
      fprintf(fstats, "\n");
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " subinterval %3d: [%.4e , %.4e]\n", sl, ai, bi);
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " polynomial [type %d] deg %d, bar %e gam %e\n", pol.type, pol.deg, pol.bar, pol.gam);
    }
    //-------------------- then call ChenLanNr    
    ierr = pEVSL_ChebLanNr(pevsl, xintv, mlan, tol, &vinit, &pol, &nev2, &lam, &Y, &res, fstats);
    if (ierr) {
      printf("ChebLanNr error %d\n", ierr);
      return 1;
    }
    /*-------------------- output results */
    /* sort the eigenvals: ascending order
     * ind: keep the orginal indices */
    ind = (int *) malloc(nev2*sizeof(int));
    sort_double(nev2, lam, ind);
    /* group leader checks the eigenvalues and print */
    if (comm.group_rank == 0) {
      /* compute exact eigenvalues */
      ExactEigLap3(nx, ny, nz, ai, bi, &nev_ex, &lam_ex);
      fprintf(fstats, " [Group %d]: number of eigenvalues: %d, found: %d\n", 
              comm.group_id, nev_ex, nev2);

      sprintf(msg+strlen(msg), " ======================================================\n");
      sprintf(msg+strlen(msg), " subinterval %3d: [%.4e , %.4e]\n", sl, ai, bi);
      sprintf(msg+strlen(msg), " [Group %d]: number of eigenvalues: %d, found: %d\n",
              comm.group_id, nev_ex, nev2);

      /* print eigenvalues */
      fprintf(fstats, "                                   Eigenvalues in [a, b]\n");
      fprintf(fstats, "     Computed [%d]       ||Res||              Exact [%d]", nev2, nev_ex);
      if (nev2 == nev_ex) {
        fprintf(fstats, "                 Err");
      }
      fprintf(fstats, "\n");
      for (i=0; i<max(nev2, nev_ex); i++) {
        if (i < nev2) {
          fprintf(fstats, "% .15e  %.1e", lam[i], res[ind[i]]);
        } else {
          fprintf(fstats, "                               ");
        }
        if (i < nev_ex) { 
          fprintf(fstats, "        % .15e", lam_ex[i]);
        }
        if (nev2 == nev_ex) {
          fprintf(fstats, "        % .1e", lam[i]-lam_ex[i]);
        }
        fprintf(fstats,"\n");
        if (i>50) {
          fprintf(fstats,"                        -- More not shown --\n");
          break;
        } 
      }
      free(lam_ex);
    }
    /*-------------------- free within this slice */
    if (lam) { free(lam); }
    if (Y) {
      pEVSL_ParvecsFree(Y);
      free(Y);
    }
    if (res) { free(res); }
    pEVSL_FreePol(&pol);
    free(ind);
    /*--------------------- print stats */
    pEVSL_StatsPrint(pevsl, fstats);
  } /* for (sl=0 */

  /* group leaders print on screen in orders */
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    fprintf(stdout, "%s", msg);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  /*--------------------- done */
  if (msg) {
    free(msg);
  }
  if (fstats) {
    fclose(fstats);
  }
  free(sli);
  free(mu);

  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&vinit);
  CommInfoFree(&comm);
 
  pEVSL_Finish(pevsl);
  MPI_Finalize();

  return 0;
}

